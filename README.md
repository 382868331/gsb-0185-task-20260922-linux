# csg_rays — 带空腔与可复用实例的 CSG 射线区间求交

离线 CAD 几何工程用可复用模块：对 CSG 场景（叶为球与轴对齐盒，内部节点
union / intersection / difference，实例叶 `Instance` 引用已有子树并施加
**可逆仿射变换**）与一条射线求交，输出排序不相交的实体 t 区间、零测度
相切接触、带进入/退出面来源的实体边界，以及最近正向实体边界。

同一棵子树可以被多个实例引用、摆到不同位置：变换只属于这次引用，共享
节点不被复制或修改；引用边不得成环。

* **Python 版本**：3.14.7（仅用标准库：`fractions` / `decimal` / `math` /
  `dataclasses` / `functools`）。
* **依赖**：无任何第三方依赖，安装 Python 后即可运行；核心、测试、演示
  全部离线，不需要网络、账号、数据库或 Docker。
* **平台**：Linux 原生（任何支持 CPython 3.14.7 的平台均可）。

```bash
python -m unittest discover -s tests -v   # 运行测试
python demo.py                            # 运行演示（约 0.01 秒内完成）
```

## 接口

```python
from csg_rays import (
    Sphere, Box, Union, Intersection, Difference, Instance, Ray,
    ray_intervals,
    RayIntervalsResult, Interval, Touch, Boundary, Source,
    CSGError, GeometryError, CertificationBudgetExhausted,
    MAX_TREE_NODES, MAX_LEAF_OCCURRENCES,
)

# 可复用子树：一个单位球
ball = Sphere((0, 0, 0), 1, "ball")

# 同一子树的两次引用：原位 + 缩放到 2 倍并平移到 (10,0,0)
scene = Union([
    Instance(ball, [[1,0,0],[0,1,0],[0,0,1]], (0, 0, 0), "orig"),
    Instance(ball, [[2,0,0],[0,2,0],[0,0,2]], (10, 0, 0), "big"),
])
res = ray_intervals(scene, Ray((-5, 0, 0), (1, 0, 0)))
# 两次引用分别给出区间 (-1,1) 与 (9,11)，来源用 path 区分
```

### 树节点

| 构造 | 说明 |
| --- | --- |
| `Sphere(center, radius, leaf_id)` | 实体球叶。`radius > 0`。 |
| `Box(mins, maxs, leaf_id)` | 轴对齐实体盒叶。每轴 `maxs[i] > mins[i]`。 |
| `Union(children)` / `Intersection(children)` | n 元并 / 交，`children` 非空可迭代。 |
| `Difference(positive, *subtractors)` | 正体减去一个或多个减体（正则化差，减体按闭实体处理）。 |
| `Instance(child, linear, translation=(0,0,0), instance_id="instance")` | 对已有子树 `child` 的一次引用，施加可逆仿射映射 `p_world = A @ p_local + b`；`linear` 为 3×3 矩阵（行序列），`translation` 为 3 维向量。变换只作用于本次引用，`child` 不会被修改，可被多个实例共享。 |
| `Ray(origin, direction)` | 射线 `origin + t*direction`，`t ∈ ℝ`（整条直线）；`direction` 必须非零。 |

数值字段接受 `int` 或有限 `float`，内部一律转为精确的
`fractions.Fraction` 保存。`leaf_id` 必须是非空 `str`，且在所有**不同的
叶节点**间唯一；`instance_id` 必须是非空 `str`，且在所有不同的实例节点
间唯一（同一节点被多次引用不算重复）。

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
  按 `(path, leaf_id, face)` 字典序排序；`source` 为代表（排序第一）。
* `Source`：
  * `leaf_id` — 原叶 ID（共享叶被引用多次时相同）；
  * `path` — 从场景根到该次引用经过的 `instance_id` 元组（未实例化的叶
    为 `()`）；`(path, leaf_id)` 唯一区分一次具体引用；
  * `face` — 球为 `"sphere"`；盒为
    `"x_lo"/"x_hi"/"y_lo"/"y_hi"/"z_lo"/"z_hi"`（局部坐标下的面名）；
  * `kind` — 叶侧事件 `"enter"/"exit"/"touch"`；
  * `normal` — 命中点处的**世界坐标**单位外法向（3 元 float）：局部面法
    向经实例矩阵的逆转置变换并归一化；处于 `Difference` 减体一侧时
    **翻转**且 `flipped=True`，嵌套差按奇偶累计。

## 实例与变换的语义约定

* **射线逆变换**：世界射线 `p(t) = o + t·d` 经某次引用的帧 `A, b` 逆变换
  到叶的局部坐标：`o_local = A⁻¹(o − b)`，`d_local = A⁻¹d`。方向**不归一
  化**：因为 `o + t·d = A(o_local + t·d_local) + b`，命中参数仍是同一个
  世界 t，区间端点与 t 排序都不因缩放/旋转/镜像而错位。
* **负行列式（镜像）**：`d_local` 可以反方向，但求交只解参数方程、从不
  依赖"射线朝哪边走"，因此**不会把进入/退出简单互换成错误区间**；镜像
  实体的外法向由逆转置公式自然给出（手性反转），t 顺序仍按世界方向
  报告 enter→exit。
* **法向逆转置**：切平面法向按 `n_world = normalize(A⁻ᵀ n_local)` 变换
  （各嵌套实例的矩阵先精确相乘复合，再求一次逆），缩放球得到椭球的法
  向、剪切盒面的倾斜法向均由此得到；仅做一次归一化舍入。
* **difference 翻转层级**：极性只在 `Difference` 的减体分支取反，实例节点
  本身不改变极性；嵌套实例位于减体子树内时，其来源法向照样翻转，且翻
  转发生在正确的布尔层级。
* **来源去重**：两个引用即便几何重合，也是 `path` 不同的两个来源；同一
  t 的所有事件仍按既有同 t 批量规则一次更新，不引起状态改变的重合事件
  不产生伪边界，完全重合的两次引用做差仍为空。
* **正则化闭实体 / touch 规则**：变换不改变拓扑，切线接触、贴面接触等
  零测度集合的吞并与报告规则与无实例时完全一致。
* **射线是整条直线**：区间覆盖所有 `t ∈ ℝ`；`nearest_boundary` 额外筛选
  `t > 0`。

## 输入上限与错误语义

* 场景中**不同的已声明节点**总数（叶 + 内部节点 + 实例节点，按对象去
  重）≤ `MAX_TREE_NODES = 100`。实例允许子树共享：同一对象经多个
  `Instance` 引用只计一次节点，但展开后的**叶出现次数** ≤
  `MAX_LEAF_OCCURRENCES = 2000`（嵌套引用可能指数膨胀，此为独立的工作量
  上限）。
* 共享只允许通过 `Instance` 引用：非实例边（union/intersection/
  difference 的子边）必须构成树，同一对象出现两次报
  `GeometryError`；沿实例引用边回到当前遍历栈上的节点即**引用环**，报
  `GeometryError`（实例节点使用槽属性，构造后改出环也会被检出）。
* 实例矩阵必须**可逆**：行列式为 0（奇异矩阵，含镜像复合成降秩的情形）
  在构造 `Instance` 时即拒绝；矩阵必须恰为 3×3、平移恰为 3 个数值。
* 数值输入拒绝 `bool`、NaN、Infinity；球半径 `> 0`；盒
  `maxs[i] > mins[i]`；射线方向非零。
* 非法输入抛出 `GeometryError`，消息含可定位路径（如
  `Instance(instance_id='m').linear[1][2]: ...`、
  `tree.children[0].child[Instance 'm']: ...`），不做任何静默纠正。
  `ray_intervals` 为纯函数：校验失败时不产生任何部分结果或副作用。
* `CertificationBudgetExhausted`（`CSGError` 子类）与"证明无交"严格分
  开：无交返回空结果；只有当两个**不同**事件参数接近到 240 位 Decimal
  预算内仍无法证明其先后顺序时才抛出（小整数系数下实际不会触发）。

## 浮点误差口径

* 所有事件参数在内部精确表示：盒事件、有理球根、切点均为
  `Fraction`；球的非有理根为精确二次根对象（Fraction 系数 + 根号符
  号）。仿射复合、矩阵求逆全部是 Fraction 精确运算。
* 重合判定（同 t 聚类）是**精确**的：Fraction 直接比较；根与有理数由
  二次式符号精确判定；根与根先精确判等（同最小多项式），不同根的次序
  用 60→120→240 位 Decimal 升级认证，超预算则抛
  `CertificationBudgetExhausted`，绝不靠宽容差猜测。
* 仅在输出边界做一次舍入：`t` 经 `float()` 转换，有理端点误差 ≤ 1
  ulp；无理端点先取 80 位 Decimal 再转 float，相对误差 ≲ 1e-15。
* 法向：未变换的盒面法向精确；局部球法向在有理 t 时精确（Fraction），
  无理 t 时用 70 位 Decimal。经实例变换后，逆转置结果为 Fraction 时若其
  长度是有理完全平方则精确归一化，否则以 70 位 Decimal 归一化，相对误
  差 ≲ 1e-15；`t` 与法向的独立核对（代回隐式方程、与解析梯度比较）见
  `tests/test_instances.py`。
* 边界之间不存在"宽容差"：区间不相交、不重叠由精确排序保证。

## 设计取舍

* **DAG 展开为叶出现，扫描算法不变**：`_prepare` 沿引用边把场景展开成
  带（极性、实例路径、复合仿射帧）的叶出现列表和一棵布尔求值树；之后
  的三值扫描、同 t 批量、正则化、来源判定完全复用原有数据流，没有并
  行的第二套实现。共享节点只被遍历多次，从不复制。
* **三值状态扫描**：每叶在每段开区间上取 IN / T / OUT 三态，布尔运算按
  三态真值表求值（减体为闭集），再对 T 段做正则化归并。
* **精确算术优先**：矩阵、射线逆变换、事件参数、重合聚类全程 Fraction；
  Decimal 仅用于无理根次序认证和个别无理法向归一化。

## 已知限制

* 叶仅球与轴对齐盒；实例可施加任意**可逆**仿射变换（由此可得到椭球、
  旋转/剪切盒、镜像体等），但不存在其他图元；布尔算子仅 union /
  intersection / difference。这是本题的数学子集，不宣称完整 CAD 内核
  或行业标准兼容。
* 奇异（降秩）实例被拒绝：退化映射把实体压成面/线/点，不属于本题契
  约。
* 节点上限按不同声明节点计 100；实例展开的叶出现上限 2000 为接口边
  界，短演示不穷举。
* 输出 t 与法向为 float（误差口径见上）；需要完全精确结果时可基于节点
  中保存的 Fraction 输入自行重算。
* `CertificationBudgetExhausted` 的触发路径（相异根间距 < 1e-230 量
  级）无对应单元测试——用小整数系数无法构造。
* 结果为整条直线上的区间；若只需 `t ≥ 0` 的射线段，请自行过滤
  （`nearest_boundary` 已按 `t > 0` 过滤）。

## 仓库结构

```
csg_rays/           # 库本体（纯标准库）
  __init__.py       # 公开接口导出
  _types.py         # 数据结构、输入校验、Instance 节点、错误类型
  _exact.py         # Fraction/二次根精确比较与 Decimal 升级认证
  _core.py          # DAG 校验展开、叶事件、三值扫描、区间/边界装配
tests/
  test_analytic.py    # 解析精确端点、重合盒面、切线、内部起射、错误语义
  test_instances.py   # 缩放球 vs 解析椭球、剪切盒、镜像差集、嵌套/共享
                      # 实例、奇异矩阵与引用环（含隐式方程/法向独立核对）
  test_sampling.py    # 独立参考分类器的小采样（含随机仿射实例场景、
                      # 独立法向方向核对；不调用被测核心生成期望值）
demo.py             # 固定输入演示：同一子树两次摆放 + 镜像 + 奇异矩阵拒绝
```
