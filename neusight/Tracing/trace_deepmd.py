"""
DeepMD-kit 推理算子图构造器 (v2 — 源码级精确建模)

将 DeepMD-kit 的推理过程"解析式"地分解为 NeuSight 兼容的算子序列
（Linear / BMM / VEC* / MEM），无需 HuggingFace FX tracing。

v2 变更 (基于 DeepMD-kit 源码分析):
  - 修复 embedding net: 按 type_one_side 生成 ntypes 个独立子网络
  - 修复 descriptor BMM: 循环内 matmul(rr.T, gg) 和循环外 matmul(G1, G2)
  - 修复 resnet skip: 区分 same_dim(add) vs double_dim(concat+add)
  - 修复 force backward: 使用 per-type embedding backward

支持的 descriptor 类型:
  - se_e2_a  (标准两体 smooth edition)
  - se_atten (DPA-1, attention-based)

参考源码:
  - se_a.py: DescrptBlockSeA.forward() L737-853
  - env_mat.py: _make_env_mat() L11-48
  - mlp.py: MLPLayer.forward() L188-219
  - fitting.py: GeneralFitting._forward_common() L739-857
"""

import pandas as pd


def _op(name, opname, fw_ops, input_shapes, output_shape):
    """构造一行算子记录，格式与 parse_trace() 输出一致"""
    return {
        "Name": name,
        "OpName": opname,
        "FwOps": fw_ops,
        "BwOps": [],
        "AccOps": [],
        "InputShapes": input_shapes,
        "OutputShape": output_shape,
    }


# ============================================================
# 阶段 1: Neighbor List (未建模 — 归入 overhead model)
# ============================================================
def _build_neighbor_list_ops(N, M):
    """
    Neighbor list 的建模算子 (仅代表最终输出的 gather 结果)。

    实际的 O(N²) broadcast diff + topk 在 overhead model 中用
    C_QUAD × N × nall × 8 / bw 建模，这里只建模筛选后的 N×M 输出。

    源码: nlist.py build_neighbor_list() L47-135
    """
    NM = N * M
    ops = []

    # 最终的 neighbor coordinates gather (筛选后的 N×M 个邻居)
    ops.append(_op(
        "nlist_gather", "MEM",
        [("MEM", [(NM, 3)])],
        [(NM, 3)],
        (NM, 3),
    ))

    return ops


# ============================================================
# 阶段 2: Environment Matrix
# ============================================================
def _build_env_matrix_ops(N, M):
    """
    环境矩阵构造。

    源码: env_mat.py _make_env_mat() L11-48
      coord_r = torch.gather(coord_pad, 1, index)      # gather 邻居坐标
      diff = coord_r - coord_l                           # 坐标差
      length = torch.linalg.norm(diff, dim=-1)           # 距离
      t0 = 1 / length                                    # 1/r
      t1 = diff / length^2                               # x/r², y/r², z/r²
      weight = compute_smooth_weight(length, ...)         # 5阶多项式
      env_mat = cat([t0, t1], dim=-1) * weight            # [N, M, 4]
    """
    NM = N * M
    ops = []

    # gather neighbor coordinates: [N, M, 3]
    ops.append(_op(
        "env_gather", "MEM",
        [("MEM", [(N, M, 3)])],
        [(N, M, 3)],
        (N, M, 3),
    ))

    # diff + norm: coord_r - coord_l, then linalg.norm
    ops.append(_op(
        "env_diff_norm", "VECadd",
        [("VECadd", (NM, 3))],
        [(NM, 3), (NM, 3)],
        (NM, 3),
    ))

    # 1/r, x/r², y/r², z/r² — division + square
    ops.append(_op(
        "env_inv_r", "VECmul",
        [("VECmul", (NM, 4))],
        [(NM, 4)],
        (NM, 4),
    ))

    # smooth weight: 5阶多项式 (uu^3 * (-6uu^2 + 15uu - 10) + 1)
    ops.append(_op(
        "env_smooth_weight", "VECmul",
        [("VECmul", (NM, 1))],
        [(NM, 1)],
        (NM, 1),
    ))

    # env_mat = cat([t0, t1]) * weight — element-wise multiply
    ops.append(_op(
        "env_mat_mul", "VECmul",
        [("VECmul", (NM, 4))],
        [(NM, 4), (NM, 1)],
        (NM, 4),
    ))

    # normalization: (env_mat - mean) / stddev
    ops.append(_op(
        "env_normalize", "VECadd",
        [("VECadd", (NM, 4))],
        [(NM, 4), (NM, 4)],
        (NM, 4),
    ))

    return ops


# ============================================================
# 阶段 3: Embedding Network (per-type)
# ============================================================
def _build_single_embedding_net_ops(prefix, batch, neuron, activation="tanh"):
    """
    构建单个 embedding network 的 MLP 算子序列。

    源码: mlp.py MLPLayer.forward() L188-219
      yy = F.linear(xx, weight, bias)
      yy = activate(yy)
      yy = yy * idt  (if use_timestep, i.e. resnet_dt=True)
      if resnet:
          if xx.shape[-1] == yy.shape[-1]:
              yy = yy + xx
          elif 2 * xx.shape[-1] == yy.shape[-1]:
              yy = yy + concat([xx, xx])

    Parameters
    ----------
    prefix : str
        算子名称前缀 (e.g. "emb_t0")
    batch : int
        batch dimension (N * sel_i for per-type embedding)
    neuron : list[int]
        网络各层维度 [25, 50, 100]
    activation : str
        激活函数名
    """
    act_name = f"VEC{activation}"
    ops = []

    in_dim = 1  # 输入: 标量 s(r)
    for i, out_dim in enumerate(neuron):
        # Linear: (batch, in_dim) -> (batch, out_dim)
        ops.append(_op(
            f"{prefix}_linear_{i}", "Linear",
            [("Linear", (batch, in_dim, out_dim))],
            [(batch, in_dim)],
            (batch, out_dim),
        ))

        # Activation: tanh / gelu / etc.
        ops.append(_op(
            f"{prefix}_act_{i}", act_name,
            [(act_name, (batch, out_dim))],
            [(batch, out_dim)],
            (batch, out_dim),
        ))

        # ResNet skip connection
        if i > 0:
            if neuron[i] == neuron[i - 1]:
                # same dim: yy = yy + xx
                ops.append(_op(
                    f"{prefix}_res_{i}", "VECadd",
                    [("VECadd", (batch, out_dim))],
                    [(batch, out_dim), (batch, out_dim)],
                    (batch, out_dim),
                ))
            elif neuron[i] == 2 * neuron[i - 1]:
                # double dim: yy = yy + concat([xx, xx])
                # concat 本身是 MEM op, add 是 VECadd
                ops.append(_op(
                    f"{prefix}_concat_{i}", "MEM",
                    [("MEM", [(batch, out_dim)])],
                    [(batch, neuron[i - 1])],
                    (batch, out_dim),
                ))
                ops.append(_op(
                    f"{prefix}_res_{i}", "VECadd",
                    [("VECadd", (batch, out_dim))],
                    [(batch, out_dim), (batch, out_dim)],
                    (batch, out_dim),
                ))

        in_dim = out_dim

    return ops


def _build_embedding_net_ops(N, sel, neuron, activation="tanh",
                              type_one_side=True):
    """
    阶段3: Embedding Network — 按 atom type 独立构建。

    源码: se_a.py DescrptBlockSeA.forward() L789-836
      type_one_side=True 时:
        for ii in range(ntypes):  # 按邻居类型分
            rr = dmatrix[:, sec[ii]:sec[ii+1], :]  # 取 sel[ii] 个邻居
            ss = rr[:, :, :1]                       # 标量输入
            gg = filter_layers[ii].forward(ss)       # 独立的 embedding net
            gr = matmul(rr.T, gg)                    # [N, 4, ng]
            xyz_scatter += gr                        # 累加

    Parameters
    ----------
    N : int
        原子数
    sel : list[int]
        每种 type 的最大邻居数, e.g. [46, 92]
    neuron : list[int]
        embedding net 各层维度, e.g. [25, 50, 100]
    activation : str
        激活函数
    type_one_side : bool
        如果 True, ntypes 个独立网络; 如果 False, ntypes^2 个
    """
    ntypes = len(sel)
    ng = neuron[-1]  # embedding output dim
    ops = []

    if type_one_side:
        # ntypes 个独立 embedding net, 每个处理 sel[i] 个邻居
        for ti in range(ntypes):
            ni = sel[ti]  # 该类型的邻居数
            batch = N * ni

            # 单个 type 的 embedding net
            ops.extend(_build_single_embedding_net_ops(
                f"emb_t{ti}", batch, neuron, activation
            ))

            # matmul: rr.T @ gg = [N, 4, ni] x [N, ni, ng] -> [N, 4, ng]
            # 源码: gr = torch.matmul(rr.permute(0,2,1), gg)
            ops.append(_op(
                f"emb_t{ti}_matmul", "BMM",
                [("BMM", (N, 4, ni, ng))],
                [(N, ni, 4), (N, ni, ng)],
                (N, 4, ng),
            ))

            # 累加: xyz_scatter += gr
            if ti > 0:
                ops.append(_op(
                    f"emb_t{ti}_accum", "VECadd",
                    [("VECadd", (N, 4 * ng))],
                    [(N, 4 * ng), (N, 4 * ng)],
                    (N, 4 * ng),
                ))
    else:
        # ntypes^2 个独立 embedding net (type_one_side=False)
        # 源码: se_a.py L789-836 双层循环 (ti=center, tj=neighbor)
        #   for ti in range(ntypes):
        #     for tj in range(ntypes):
        #       rr = dmatrix[mask_ti, sec[tj]:sec[tj+1], :]
        #       gg = filter_layers[ti][tj].forward(rr[:,:,:1])
        #       gr = matmul(rr.permute(0,2,1), gg)
        #       xyz_scatter[mask_ti] += gr
        #
        # 实际 launch 时, 由于 mask + scatter 操作, 每个 (ti,tj) 子网仍按
        # full N × sel[tj] 启动 (中心原子掩码在 GPU 上实现)。
        # 这里按 full batch 建模, 与真实 kernel launch 数对齐。
        accum_idx = 0
        for ti in range(ntypes):
            for tj in range(ntypes):
                ni = sel[tj]
                batch = N * ni

                # (ti, tj) 子网 — 独立的 MLP filter
                ops.extend(_build_single_embedding_net_ops(
                    f"emb_t{ti}_n{tj}", batch, neuron, activation
                ))

                # per-pair matmul: [N, 4, ni] @ [N, ni, ng] -> [N, 4, ng]
                ops.append(_op(
                    f"emb_t{ti}_n{tj}_matmul", "BMM",
                    [("BMM", (N, 4, ni, ng))],
                    [(N, ni, 4), (N, ni, ng)],
                    (N, 4, ng),
                ))

                # 累加 (除第一个外)
                if accum_idx > 0:
                    ops.append(_op(
                        f"emb_t{ti}_n{tj}_accum", "VECadd",
                        [("VECadd", (N, 4 * ng))],
                        [(N, 4 * ng), (N, 4 * ng)],
                        (N, 4 * ng),
                    ))
                accum_idx += 1

    return ops


# ============================================================
# 阶段 4: Descriptor 矩阵运算
# ============================================================
def _build_descriptor_ops(N, sel, emb_out_dim, axis_neuron, type_one_side=True):
    """
    Descriptor 最终矩阵运算。

    源码: se_a.py L838-844
      xyz_scatter /= nnei                          # 归一化
      xyz_scatter_1 = xyz_scatter.permute(0,2,1)   # [N, ng, 4]
      xyz_scatter_2 = xyz_scatter[:, :, :axis_n]   # [N, 4, axis_neuron]
      result = matmul(xyz_scatter_1, xyz_scatter_2) # [N, ng, axis_neuron]

    注: per-type matmul(rr.T, gg) 已在 embedding 阶段建模。
    """
    ng = emb_out_dim
    ops = []

    # div by nnei (归一化)
    M = sum(sel)
    ops.append(_op(
        "desc_normalize", "VECmul",
        [("VECmul", (N, 4 * ng))],
        [(N, 4 * ng)],
        (N, 4 * ng),
    ))

    # 最终 matmul: xyz_scatter_1 @ xyz_scatter_2
    # [N, ng, 4] @ [N, 4, axis_neuron] -> [N, ng, axis_neuron]
    ops.append(_op(
        "desc_final_matmul", "BMM",
        [("BMM", (N, ng, 4, axis_neuron))],
        [(N, ng, 4), (N, 4, axis_neuron)],
        (N, ng, axis_neuron),
    ))

    # reshape: (N, ng, axis_neuron) -> (N, ng * axis_neuron)
    desc_dim = ng * axis_neuron
    ops.append(_op(
        "desc_reshape", "MEM",
        [("MEM", [(N, desc_dim)])],
        [(N, ng, axis_neuron)],
        (N, desc_dim),
    ))

    return ops, desc_dim


# ============================================================
# 阶段 5: Fitting Network
# ============================================================
def _build_fitting_net_ops(N, desc_dim, neuron, activation="tanh",
                            mixed_types=True, ntypes=1):
    """
    Fitting Network。

    源码: fitting.py GeneralFitting._forward_common() L847-857
      mixed_types=True:  单个网络处理所有 atoms
      mixed_types=False: ntypes 个网络, 用 mask 分开

    对 mixed_types=True (最常用), 只有 1 个网络。
    对 mixed_types=False, 生成 ntypes 个独立网络 (每个处理全部 N atoms,
    用 mask 选择)，实际计算量是 ntypes 倍。
    """
    act_name = f"VEC{activation}"
    ops = []

    n_nets = 1 if mixed_types else ntypes

    for net_idx in range(n_nets):
        prefix = "fit" if n_nets == 1 else f"fit_t{net_idx}"

        in_dim = desc_dim
        for i, out_dim in enumerate(neuron):
            # Linear
            ops.append(_op(
                f"{prefix}_linear_{i}", "Linear",
                [("Linear", (N, in_dim, out_dim))],
                [(N, in_dim)],
                (N, out_dim),
            ))

            # Activation
            ops.append(_op(
                f"{prefix}_act_{i}", act_name,
                [(act_name, (N, out_dim))],
                [(N, out_dim)],
                (N, out_dim),
            ))

            # ResNet skip connection
            if i > 0:
                if neuron[i] == neuron[i - 1]:
                    ops.append(_op(
                        f"{prefix}_res_{i}", "VECadd",
                        [("VECadd", (N, out_dim))],
                        [(N, out_dim), (N, out_dim)],
                        (N, out_dim),
                    ))
                elif neuron[i] == 2 * neuron[i - 1]:
                    ops.append(_op(
                        f"{prefix}_concat_{i}", "MEM",
                        [("MEM", [(N, out_dim)])],
                        [(N, neuron[i - 1])],
                        (N, out_dim),
                    ))
                    ops.append(_op(
                        f"{prefix}_res_{i}", "VECadd",
                        [("VECadd", (N, out_dim))],
                        [(N, out_dim), (N, out_dim)],
                        (N, out_dim),
                    ))

            in_dim = out_dim

    return ops


# ============================================================
# 阶段 6: Output
# ============================================================
def _build_output_ops(N, fit_out_dim):
    """
    Output layer: per-atom energy -> total energy

    源码: fitting.py L856
      outs = atom_property + bias_atom_e[atype]
    """
    ops = []

    # output linear: (N, fit_out_dim) -> (N, 1)
    ops.append(_op(
        "output_linear", "Linear",
        [("Linear", (N, fit_out_dim, 1))],
        [(N, fit_out_dim)],
        (N, 1),
    ))

    # sum reduction over atoms
    ops.append(_op(
        "energy_sum", "VECadd",
        [("VECadd", (N, 1))],
        [(N, 1)],
        (1,),
    ))

    return ops


# ============================================================
# 阶段 7: Force backward (autograd)
# ============================================================
def _build_force_backward_ops(N, sel, emb_neuron, fit_neuron, desc_dim,
                               type_one_side=True, mixed_types=True, ntypes=1):
    """
    Force 计算的 autograd backward。

    源码: transform_output.py task_deriv_one() L65-96
      extended_force = torch.autograd.grad([energy], [extended_coord], ...)
      → 反向遍历: output -> fitting -> descriptor -> embedding

    Backward of Linear(B, I, O): grad_input = Linear(B, O, I)
    """
    ng = emb_neuron[-1]
    ops = []

    # ----- Output backward -----
    ops.append(_op(
        "output_bw", "Linear",
        [("Linear", (N, 1, fit_neuron[-1]))],
        [(N, 1)],
        (N, fit_neuron[-1]),
    ))

    # ----- Fitting backward -----
    n_fit_nets = 1 if mixed_types else ntypes
    for net_idx in range(n_fit_nets):
        prefix = "fit_bw" if n_fit_nets == 1 else f"fit_t{net_idx}_bw"

        for i in range(len(fit_neuron) - 1, -1, -1):
            out_dim = fit_neuron[i]
            in_dim = desc_dim if i == 0 else fit_neuron[i - 1]

            ops.append(_op(
                f"{prefix}_input_{i}", "Linear",
                [("Linear", (N, out_dim, in_dim))],
                [(N, out_dim)],
                (N, in_dim),
            ))

            ops.append(_op(
                f"{prefix}_act_{i}", "VECmul",
                [("VECmul", (N, out_dim))],
                [(N, out_dim), (N, out_dim)],
                (N, out_dim),
            ))

    # ----- Descriptor backward -----
    # backward of final matmul: [N, ng, axis_neuron] -> [N, ng, 4]
    axis_neuron = desc_dim // ng if ng > 0 else 16
    ops.append(_op(
        "desc_bw_matmul", "BMM",
        [("BMM", (N, ng, axis_neuron, 4))],
        [(N, ng, axis_neuron)],
        (N, ng, 4),
    ))

    # ----- Embedding backward (per-type) -----
    if type_one_side:
        for ti in range(len(sel)):
            ni = sel[ti]
            batch = N * ni

            # backward of in-loop matmul: [N, 4, ng] -> [N, ni, ng]
            ops.append(_op(
                f"emb_t{ti}_bw_matmul", "BMM",
                [("BMM", (N, ni, 4, ng))],
                [(N, 4, ng)],
                (N, ni, ng),
            ))

            # backward of embedding MLP
            for i in range(len(emb_neuron) - 1, -1, -1):
                out_dim = emb_neuron[i]
                in_dim = 1 if i == 0 else emb_neuron[i - 1]

                ops.append(_op(
                    f"emb_t{ti}_bw_input_{i}", "Linear",
                    [("Linear", (batch, out_dim, in_dim))],
                    [(batch, out_dim)],
                    (batch, in_dim),
                ))

                ops.append(_op(
                    f"emb_t{ti}_bw_act_{i}", "VECmul",
                    [("VECmul", (batch, out_dim))],
                    [(batch, out_dim), (batch, out_dim)],
                    (batch, out_dim),
                ))
    else:
        # type_one_side=False: ntypes^2 个独立 backward 路径,
        # 与 forward 的 (ti, tj) 双层循环对称
        ntypes_local = len(sel)
        for ti in range(ntypes_local):
            for tj in range(ntypes_local):
                ni = sel[tj]
                batch = N * ni

                # backward of (ti,tj) matmul
                ops.append(_op(
                    f"emb_t{ti}_n{tj}_bw_matmul", "BMM",
                    [("BMM", (N, ni, 4, ng))],
                    [(N, 4, ng)],
                    (N, ni, ng),
                ))

                # backward of (ti,tj) embedding MLP
                for i in range(len(emb_neuron) - 1, -1, -1):
                    out_dim = emb_neuron[i]
                    in_dim = 1 if i == 0 else emb_neuron[i - 1]
                    ops.append(_op(
                        f"emb_t{ti}_n{tj}_bw_input_{i}", "Linear",
                        [("Linear", (batch, out_dim, in_dim))],
                        [(batch, out_dim)],
                        (batch, in_dim),
                    ))
                    ops.append(_op(
                        f"emb_t{ti}_n{tj}_bw_act_{i}", "VECmul",
                        [("VECmul", (batch, out_dim))],
                        [(batch, out_dim), (batch, out_dim)],
                        (batch, out_dim),
                    ))

    return ops


# ============================================================
# DPA-1 Attention (se_atten)
# ============================================================
def _build_attention_ops(N, M, emb_out_dim, num_heads=1, attn_layers=1):
    """
    DPA-1 (se_atten) 的 attention 层。

    源码: se_atten.py GatedAttentionLayer.forward() L1059-1140
      in_proj: Linear(embed_dim, hidden_dim*3)  → Q, K, V (chunk)
      Q @ K.T → attn_weights [N, num_heads, M, M]
      softmax(attn_weights)
      attn @ V → output [N, num_heads, M, head_dim]
      out_proj: Linear(hidden_dim, embed_dim)
      + LayerNorm + residual

    NeighborGatedAttention 有 attn_layers 层堆叠。
    """
    NM = N * M
    head_dim = emb_out_dim // num_heads
    ops = []

    for layer_idx in range(attn_layers):
        lp = f"attn_L{layer_idx}" if attn_layers > 1 else "attn"

        # in_proj: Linear(embed_dim, hidden_dim * 3) 一次性投影 Q, K, V
        ops.append(_op(
            f"{lp}_in_proj", "Linear",
            [("Linear", (NM, emb_out_dim, emb_out_dim * 3))],
            [(NM, emb_out_dim)],
            (NM, emb_out_dim * 3),
        ))

        # Q @ K^T → attention scores: [N*num_heads, M, M]
        ops.append(_op(
            f"{lp}_scores", "BMM",
            [("BMM", (N * num_heads, M, M, head_dim))],
            [(N * num_heads, M, head_dim)],
            (N * num_heads, M, M),
        ))

        # softmax
        ops.append(_op(
            f"{lp}_softmax", "VECsoftmax",
            [("VECsoftmax", (N * num_heads * M, M))],
            [(N * num_heads, M, M)],
            (N * num_heads, M, M),
        ))

        # attn @ V → [N*num_heads, M, head_dim]
        ops.append(_op(
            f"{lp}_output", "BMM",
            [("BMM", (N * num_heads, M, head_dim, M))],
            [(N * num_heads, M, M)],
            (N * num_heads, M, head_dim),
        ))

        # out_proj: Linear(hidden_dim, embed_dim)
        ops.append(_op(
            f"{lp}_out_proj", "Linear",
            [("Linear", (NM, emb_out_dim, emb_out_dim))],
            [(NM, emb_out_dim)],
            (NM, emb_out_dim),
        ))

        # residual + LayerNorm
        ops.append(_op(
            f"{lp}_residual", "VECadd",
            [("VECadd", (NM, emb_out_dim))],
            [(NM, emb_out_dim), (NM, emb_out_dim)],
            (NM, emb_out_dim),
        ))

    return ops


# ============================================================
# Main: 构建完整算子图
# ============================================================
def build_deepmd_opgraph(config, num_atoms, compute_force=False):
    """
    将 DeepMD-kit 模型的推理过程分解为 NeuSight 算子图。

    Parameters
    ----------
    config : dict
        DeepMD 模型配置，包含 descriptor / fitting_net / type_embedding 等字段
    num_atoms : int
        原子数
    compute_force : bool
        是否包含 force 计算（autograd backward）开销

    Returns
    -------
    pd.DataFrame
        列: Name, OpName, FwOps, BwOps, AccOps, InputShapes, OutputShape
        与 parse_trace() 输出格式完全兼容
    """
    N = num_atoms
    desc_cfg = config["descriptor"]
    fit_cfg = config["fitting_net"]

    model_type = config.get("model_type", desc_cfg.get("type", "se_e2_a"))
    sel = desc_cfg["sel"]
    if isinstance(sel, int):
        sel = [sel]
    M = sum(sel)  # total max neighbors
    emb_neuron = desc_cfg["neuron"]          # e.g. [25, 50, 100]
    axis_neuron = desc_cfg.get("axis_neuron", 16)
    fit_neuron = fit_cfg["neuron"]            # e.g. [240, 240, 240]
    type_one_side = desc_cfg.get("type_one_side", True)
    ntypes = len(sel)

    # fitting net: mixed_types is default True for modern DeepMD
    mixed_types = fit_cfg.get("mixed_types", True)

    # activation function
    emb_act = desc_cfg.get("activation_function", "tanh")
    fit_act = fit_cfg.get("activation_function", "tanh")

    # 构建算子序列
    all_ops = []

    # 阶段 1: Neighbor List (最终 gather 结果)
    all_ops.extend(_build_neighbor_list_ops(N, M))

    # 阶段 2: Environment Matrix + Smooth
    all_ops.extend(_build_env_matrix_ops(N, M))

    # 阶段 3: Embedding Network (per-type) + in-loop matmul
    all_ops.extend(_build_embedding_net_ops(
        N, sel, emb_neuron, emb_act, type_one_side
    ))

    # 阶段 3.5 (可选): Attention (DPA-1 / se_atten)
    if model_type in ("se_atten", "dpa1", "DPA-1"):
        attn_heads = desc_cfg.get("attn_heads", 1)
        attn_layers = desc_cfg.get("attn_layer", 2)
        all_ops.extend(_build_attention_ops(
            N, M, emb_neuron[-1], attn_heads, attn_layers
        ))

    # 阶段 4: Descriptor 最终矩阵运算
    desc_ops, desc_dim = _build_descriptor_ops(
        N, sel, emb_neuron[-1], axis_neuron, type_one_side
    )
    all_ops.extend(desc_ops)

    # 阶段 5: Fitting Network
    all_ops.extend(_build_fitting_net_ops(
        N, desc_dim, fit_neuron, fit_act, mixed_types, ntypes
    ))

    # 阶段 6: Output
    all_ops.extend(_build_output_ops(N, fit_neuron[-1]))

    # 阶段 7 (可选): Force backward
    if compute_force:
        all_ops.extend(_build_force_backward_ops(
            N, sel, emb_neuron, fit_neuron, desc_dim,
            type_one_side, mixed_types, ntypes
        ))

    # 构造 DataFrame
    df = pd.DataFrame(all_ops)
    return df


def count_op_types(df):
    """
    统计算子图中各类型 op 的数量。

    Parameters
    ----------
    df : pd.DataFrame
        build_deepmd_opgraph() 的输出

    Returns
    -------
    dict
        {OpName: count} 映射
    """
    return dict(df["OpName"].value_counts())
