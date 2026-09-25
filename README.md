# csg_rays — 带空腔与实例引用的 CSG 射线区间求交

离线 CAD 几何工程用可复用模块：对 CSG 树（叶为球与轴对齐盒，内部节点
union / intersection / difference，另有**实例叶** `Instance` 引用已有子树
并挂一个可逆仿射变换）与一条射线求交，输出排序不相交的实体 t 区间、零测
度相切接触、带进入/退出面来源的实体边界，以及最近正向实体边界。

* **Python 版本**：3.14.7（仅用标准库：`fractions` / `decimal` / `math` /
  `dataclasses` / `functools`）。
* **依赖**：无任何第三方依赖，安装 Python 后即可运行；核心、测试、演示
  全部离线，不需要网络、账号、数据库或 Docker。
* **平台**：Linux 原生（任何支持 CPython 3.14.7 的平台均可）。

```bash
./.venv/bin/python -m unittest discover -s tests -v  # 运行全部测试
./.venv/bin/python demo.py                           # 运行演示（0.1 秒内）
```

## 接口

```python
from csg_rays import (
    Sphere, Box, Union, Intersection, Difference, Instance, Ray,
    ray_intervals,
    RayIntervalsResult, Interval, Touch, Boundary, Source,
    CSGError, GeometryError, CertificationBudgetExhausted,
    MAX_TREE_NODES,
)

tree = Difference(Sphere((0, 0, 0), 5, "outer"), Sphere((0, 0, 0), 2, "inner"))
res = ray_intervals(tree, Ray((-10, 0, 0), (1, 0, 0)))
# res.intervals -> (Interval(5.0, 8.0, ...), Interval(12.0, 15.0, ...))
# res.nearest_boundary -> Boundary(t=5.0, kind="enter", ...)

# 同一个球子树复用为两个位置（缩放椭球 + 平移副本），共享节点不被修改
ball = Sphere((0, 0, 0), 1, "ball")
two = Union([
    Instance(ball, ((2, 0, 0), (0, 3, 0), (0, 0, 1)), (0, 0, 0), "ell"),
    Instance(ball, ((1, 0, 0), (0, 1, 0), (0, 0, 1)), (9, 0, 0), "copy"),
])
```

### 树节点

| 构造 | 说明 |
| --- | --- |
| `Sphere(center, radius, leaf_id)` | 实体球叶。`radius > 0`。 |
| `Box(mins, maxs, leaf_id)` | 轴对齐实体盒叶。每轴 `maxs[i] > mins[i]`。 |
| `Union(children)` / `Intersection(children)` | n 元并 / 交，`children` 非空可迭代。 |
| `Difference(positive, *subtractors)` | 正体减去一个或多个减体（正则化差，减体按闭实体处理）。 |
| `Instance(child, matrix, translation, instance_id)` | 对**已有** CSG 子树 `child` 的一次引用，挂可逆仿射变换 `x_world = matrix @ x_local + translation`。`matrix` 为 3×3 数值矩阵且行列式非零，`translation` 为 3 向量；同一 `child` 可被多个 Instance 共享引用，变换只作用于本次引用，不修改共享节点。禁止引用环。 |
| `Ray(origin, direction)` | 射线 `origin + t*direction`，`t ∈ ℝ`（整条直线）；`direction` 必须非零。 |

`Instance` 的矩阵与平移接受 `int`/有限 `float`，内部全部精确转为
`fractions.Fraction`；逆矩阵在构造时以伴随矩阵/行列式**精确**求出（奇异
矩阵在构造期即抛 `GeometryError`）。`instance_id` 必须为非空 `str`，同一
嵌套层级内不得重复；完整"实例路径"（从外到内的 instance_id 序列）唯一
标识一次引用。

数值字段接受 `int` 或有限 `float`，内部一律转为精确的
`fractions.Fraction` 保存（节点上的 `center`/`radius`/`mins`/`maxs` 等属性
即 Fraction 值）。`leaf_id` 必须是非空 `str`，且在整棵树内唯一。

### 输出结构

* `RayIntervalsResult.intervals: tuple[Interval, ...]` — 排序、两两不相交
  的**开**实体区间 `(t0, t1)`，`start`/`end` 为对应的 `Boundary`。
* `RayIntervalsResult.touches: tuple[Touch, ...]` — 零测度相切接触，**不**
  算实体内部：孤立切点 `t0 == t1`；射线贴盒面行进时为零厚度接触段
  `t0 < t1`。
* `RayIntervalsResult.boundaries: tuple[Boundary, ...]` — 全部实体边界，
  按 t 排序；`kind` 为 `"enter"`（空→实体）或 `"exit"`（实体→空）。
* `RayIntervalsResult.nearest_boundary` — t > 0 的最小实体边界（`Boundary`
  或 `None`）。touch 不是实体边界，不参与。
* `Boundary.sources` / `Touch.sources` — 该边界/接触的**全部**来源面，
  按 `(path, leaf_id, face)` 字典序排序；`source` 为代表，即路径+叶 ID
  字典序最小者。
* `Source`：`leaf_id`（原叶 ID）、`path`（到达该叶所经实例引用的
  `instance_id` 元组，外→内；直接放置的叶为 `()`）、`face`（球为
  `"sphere"`；盒为
  `"x_lo"/"x_hi"/"y_lo"/"y_hi"/"z_lo"/"z_hi"`）、`kind`
  （叶侧事件 `"enter"/"exit"/"touch"`）、`normal`（世界坐标命中点处
  单位外法向，3 元 float 元组；处于 `Difference` 减体一侧时**翻转**且
  `flipped=True`，嵌套差按奇偶累计）。同一原叶的两个引用以不同 `path`
  成为不同来源。

## 语义约定

* **正则化闭实体**：实体按闭集理解，输出内部为开区间。切线接触、贴面
  接触等零测度集合不产生实体内部：被实体夹住的相切事件被"吞并"（不
  在实体上戳出零宽洞），两侧皆空的相切事件报告为 `Touch`。
* **同一 t 的事件批量处理**：重合事件按同一 t 聚类后一次性更新状态；
  不引起状态改变的重合事件**不**生成伪实体边界（如两盒共面拼接的并
  只产生一个区间；完全相同两叶的差结果为空且无边界）。
* **来源判定**：叶（按"出现"计）在某边界/接触处有事件，且强制该叶为
  IN / OUT 会改变结果状态时，该叶面才计入 `sources`；因此无影响的重合
  事件不会出现在来源列表中。
* **射线是整条直线**：区间覆盖所有 `t ∈ ℝ`；`nearest_boundary` 额外筛
  选 `t > 0`（"最近正向"）。从实体/空腔内部起射时，负 t 区间同样给出。
* **实例的逆变换**：对每个叶出现，世界射线 `O + t·D` 经组合逆仿射
  `p_l = B·O + v + t·(B·D)` 变到局部；**局部方向不归一化**（保留同一
  t，缩放球的命中参数才不会漂移），事件 t 同时就是世界射线参数。
* **法向的逆转置**：局部外法向余向量经 `Bᵀ`（局部→世界映射的逆转置）
  映到世界并归一化，因此非均匀缩放、剪切、旋转下法向均正确；
  `Difference` 减体侧的翻转在布尔层级按极性奇偶组合，与几何变换独立。
* **负行列式（镜像）**：镜像只改变几何，变换后的射线参数化仍保持
  enter 在 exit 之前，**不**做进入/退出互换——互换会得到错误区间；镜像
  的外法向由逆转置自然给出。
* **同 t 批量规则跨引用保持**：同一共享子树的两个引用在同 t 重合的事
  件仍按一批处理（如两份完全重合副本的并只有一对边界，两个 `path`
  来源都保留）。

## 输入上限与错误语义

* **节点上限按展开计**：主树的每个节点 + 每次实例引用时被引用子树的
  每个节点，总数 ≤ `MAX_TREE_NODES = 100`。
* **结构规则**：每个展开根（主树或一棵被引用子树）内部必须是真树，
  同一对象在一个展开根内出现两次即报错；共享只能通过 `Instance`
  引用发生。一个节点既可直接放置又被引用是允许的（属于不同出现，
  按出现独立计状态）。引用环（直接或间接引用到自身祖先）以
  `GeometryError` 拒绝。
* 每个展开根内 `leaf_id` 唯一；跨引用的相同 `leaf_id` 合法——来源用
  `(path, leaf_id)` 区分。`instance_id` 为非空 `str`，同一嵌套路径下
  不得重复。
* 实例矩阵必须 3×3 且行列式非零；**奇异矩阵在构造 `Instance` 时即抛
  `GeometryError`**（不产生半初始化对象之外的可观察状态）。
* 数值输入拒绝 `bool`、NaN、Infinity；向量必须恰为 3 个数值；球半径
  `> 0`；盒 `maxs[i] > mins[i]`；射线方向非零。
* 非法输入抛出 `GeometryError`，消息含可定位路径（如
  `Sphere(leaf_id='s1').center[2]: ...`、
  `tree.children[0].child.matrix[1][2]: ...`），不做任何静默纠正。
  `ray_intervals` 为纯函数：输入校验失败时不产生任何部分结果或副作用。
* `CertificationBudgetExhausted`（`CSGError` 子类）与"证明无交"严格分
  开：无交返回空结果；只有当两个**不同**事件参数接近到 240 位 Decimal
  预算内仍无法证明其先后顺序时才抛出该异常（小整数系数下实际不会触
  发）。

## 浮点误差口径

* 所有事件参数在内部精确表示：盒事件、有理球根、切点均为
  `Fraction`；球的非有理根为精确二次根对象（Fraction 系数 + 根号符
  号）。仿射变换的矩阵、平移、逆矩阵全部精确为 Fraction，因此"逆变换
  后的射线"无引入近似。
* 重合判定（同 t 聚类）是**精确**的：Fraction 直接比较；根与有理数由
  二次式符号精确判定；根与根先精确判等（同最小多项式），不同根的次
  序用 60→120→240 位 Decimal 升级认证，超预算则抛
  `CertificationBudgetExhausted`，绝不靠宽容差猜测。
* 仅在输出边界做一次舍入：`t` 经 `float()` 转换，有理端点误差 ≤ 1
  ulp；无理端点先取 80 位 Decimal 再转 float，相对误差 ≲ 1e-15。
* 法向：盒面法向（含剪切/镜像后的逆转置与归一化）一次 `math.sqrt`
  舍入；球法向在命中点精确计算（有理 t）或经 70 位 Decimal（无理
  t），相对误差 ≲ 1e-15。测试以独立数值梯度（变换后隐式面的中心差
  分）复核方向，点积 > 0.9999。
* 因此边界之间不存在"宽容差"：区间不相交、不重叠是由精确排序保证的，
  不是靠 epsilon 合并。

## 设计取舍

* **三值状态扫描**：每个叶出现 在每段开区间上取 IN / T（零测度接触）
  / OUT 三态，布尔运算按三态真值表求值（减体为闭集），再对 T 段做正
  则化归并。这样切线、贴面、共点等退化情形与常规情形走同一条代码
  路径。
* **展开而不改写**：实例在求交时展开为"有效叶出现"（叶 + 组合逆变换
  + 极性 + 实例路径），共享节点只读复用；不复制、不修改共享子树，也
  不另建第二套求交实现。
* **逆变换射线 + 逆转置法向**：把变换全部吸收进射线与法向，叶事件生
  成、布尔扫描、touch/零测度规则完全复用；负行列式不特殊处理。
* **精确算术优先**：小整数系数下全部比较都是精确的 Fraction 运算，
  Decimal 仅用于无理根次序的认证升级，兼顾正确性与性能（100 节点规
  模单次求交为亚毫秒至毫秒级）。
* **来源代表与全集**：`sources` 保留全部来源面（含实例路径），代表按
  `(path, leaf_id, face)` 字典序确定性地选出，便于确定性下游处理。

## 已知限制

* 叶仅球与轴对齐盒；仿射实例可产生椭球、平行六面体（剪切盒）、镜像
  与任意线性组合的像，但不支持其他图元（锥、环面等）；布尔算子仅
  union / intersection / difference。这是本题的数学子集，不宣称完整
  CAD 内核或行业标准兼容。
* 被引用子树本身只在其自身局部坐标内描述（其内部盒仍轴对齐）；一切
  位姿通过外层 `Instance` 的矩阵表达。
* 共享只允许通过实例引用；一个展开根内的菱形结构（同父两分支指向同
  一对象）被拒绝。
* 节点数上限 100（按展开出现计数）为接口边界。
* 输出 t 与法向为 float（误差口径见上）；需要完全精确结果时可基于节
  点中的 Fraction 输入自行重算。
* `CertificationBudgetExhausted` 的触发路径（相异根间距 < 1e-230 量
  级）无对应单元测试——用小整数系数无法构造。
* 结果为整条直线上的区间；若只需 `t ≥ 0` 的射线段，请自行过滤（
  `nearest_boundary` 已按 `t > 0` 过滤）。

## 仓库结构

```
csg_rays/           # 库本体（纯标准库）
  __init__.py       # 公开接口导出
  _types.py         # 数据结构、Instance 仿射校验/精确求逆、错误类型
  _exact.py         # Fraction/二次根精确比较与 Decimal 升级认证
  _core.py          # 实例展开、逆变换射线、叶事件、三值扫描、法向装配
tests/
  test_analytic.py  # 解析精确端点（空心球）、重合盒面、切线、错误语义
  test_instances.py # 缩放球/椭球、剪切盒、镜像差集、嵌套实例、奇异/环
  test_sampling.py  # 独立参考（含仿射逆变换）的小采样成员与法向测试
demo.py             # 固定输入演示：空腔、实例复用、剪切盒、拒绝边界
```
