# csg_rays — 带空腔的 CSG 射线区间求交

离线 CAD 几何工程用可复用模块：对 CSG 树（叶仅球与轴对齐盒，内部节点
union / intersection / difference）与一条射线求交，输出排序不相交的实体
t 区间、零测度相切接触、带进入/退出面来源的实体边界，以及最近正向实体
边界。

* **Python 版本**：3.14.7（仅用标准库：`fractions` / `decimal` / `math` /
  `dataclasses` / `functools`）。
* **依赖**：无任何第三方依赖，安装 Python 后即可运行；核心、测试、演示
  全部离线，不需要网络、账号、数据库或 Docker。
* **平台**：Linux 原生（任何支持 CPython 3.14.7 的平台均可）。

```bash
python -m unittest discover -s tests -v   # 运行测试
python demo.py                            # 运行演示（约 0.1 秒内完成）
```

## 接口

```python
from csg_rays import (
    Sphere, Box, Union, Intersection, Difference, Ray,
    ray_intervals,
    RayIntervalsResult, Interval, Touch, Boundary, Source,
    CSGError, GeometryError, CertificationBudgetExhausted,
    ORIENT_FRONT, ORIENT_BACK, ORIENT_GRAZING,
    MAX_TREE_NODES,
)

tree = Difference(Sphere((0, 0, 0), 5, "outer", "steel"),
                  Sphere((0, 0, 0), 2, "inner", "air"))
res = ray_intervals(tree, Ray((-10, 0, 0), (1, 0, 0)))
# res.intervals -> (Interval(5.0, 8.0, ...), Interval(12.0, 15.0, ...))
# res.nearest_boundary -> Boundary(t=5.0, kind="enter", ...)
```

### 树节点

| 构造 | 说明 |
| --- | --- |
| `Sphere(center, radius, leaf_id, material=None)` | 实体球叶。`radius > 0`。 |
| `Box(mins, maxs, leaf_id, material=None)` | 轴对齐实体盒叶。每轴 `maxs[i] > mins[i]`。 |
| `Union(children)` / `Intersection(children)` | n 元并 / 交，`children` 非空可迭代。 |
| `Difference(positive, *subtractors)` | 正体减去一个或多个减体（正则化差，减体按闭实体处理）。 |
| `Ray(origin, direction)` | 射线 `origin + t*direction`，`t ∈ ℝ`（整条直线）；`direction` 必须非零。 |

数值字段接受 `int` 或有限 `float`，内部一律转为精确的
`fractions.Fraction` 保存（节点上的 `center`/`radius`/`mins`/`maxs` 等属性
即 Fraction 值）。`leaf_id` 必须是非空 `str`，且在整棵树内唯一。
`material` 可省略（为 `None`）或给定非空 `str` 作为该叶的材料 ID，原样
透传到该叶产生的每个 `Source`。

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
  按 `(leaf_id, face)` 字典序排序；`source` 为代表，即叶 ID 字典序最小者。
* `Source`：`leaf_id`、`face`（球为 `"sphere"`；盒为
  `"x_lo"/"x_hi"/"y_lo"/"y_hi"/"z_lo"/"z_hi"`）、`kind`
  （叶侧事件 `"enter"/"exit"/"touch"`）、`normal`（命中点处单位外法向，
  3 元 float 元组；处于 `Difference` 减体一侧时**翻转**且
  `flipped=True`，嵌套差按奇偶累计）、`material`（来源叶的材料 ID，未指定
  为 `None`）、`orientation`（按**叶自身几何外法向**相对射线方向分类，
  不受翻转影响）。

### 材料 ID 与朝向

* **材料 ID**：`Sphere/Box` 的 `material`（`None` 或非空 `str`）随来源面
  原样透传，布尔合成不改变它；同一实体边界若由多叶重合贡献，各 `Source`
  保留各自材料，由下游按代表或全集取用。
* **朝向 `orientation`**：在做差集法向翻转**之前**，用叶在命中点处的几何
  外法向 `n` 与射线方向 `d` 的精确点积分类：
  `n·d < 0 → "front"`（射线进入该叶所穿过的面）、
  `n·d > 0 → "back"`（退出面）、
  `n·d == 0 → "grazing"`（球切点 / 盒贴面）。
  常量为 `ORIENT_FRONT / ORIENT_BACK / ORIENT_GRAZING`。横穿事件满足
  "enter 边界必为 front、exit 边界必为 back"；差集内壁翻转的是 `normal`
  与 `flipped`，**不**改 `orientation`。点积用 Fraction 精确计算；无理球根
  的横穿由根的大小（小根进入 / 大根退出）判定，切点必为 grazing。

## 语义约定

* **正则化闭实体**：实体按闭集理解，输出内部为开区间。切线接触、贴面
  接触等零测度集合不产生实体内部：被实体夹住的相切事件被"吞并"（不
  在实体上戳出零宽洞），两侧皆空的相切事件报告为 `Touch`。
* **同一 t 的事件批量处理**：重合事件按同一 t 聚类后一次性更新状态；
  不引起状态改变的重合事件**不**生成伪实体边界（如两盒共面拼接的并
  只产生一个区间；完全相同两叶的差结果为空且无边界）。
* **重合端点的固定优先级**：同一边界由多叶重合贡献时，`sources` 全集按
  `(leaf_id, face)` 字典序排序，代表 `source` 恒取该序首位（即 leaf_id
  字典序最小），与构造顺序、算子位置无关，结果确定可复现。
* **嵌套差集的法向奇偶**：叶处于 `Difference` 减体位置的深度为奇数时
  `flipped=True`、法向翻转；为偶数（含 0）时不翻转。例如
  `A − (B − C)` 中 B 翻转、C 不翻转；`A − (B − (C − D))` 中 B、D
  翻转、C 不翻转。同深度的多个并列减体各自都翻转一次。
* **来源判定**：叶在某边界/接触处有事件，且强制该叶为 IN / OUT 会改变
  结果状态时，该叶面才计入 `sources`；因此无影响的重合事件不会出现
  在来源列表中。
* **射线是整条直线**：区间覆盖所有 `t ∈ ℝ`；`nearest_boundary` 额外筛
  选 `t > 0`（"最近正向"）。从实体/空腔内部起射时，负 t 区间同样给出。

## 输入上限与错误语义

* 树节点总数（叶 + 内部节点）≤ `MAX_TREE_NODES = 100`；树不得共享节点
  （同一对象出现两次即报错）；`leaf_id` 全局唯一。
* 数值输入拒绝 `bool`、NaN、Infinity；向量必须恰为 3 个数值；球半径
  `> 0`；盒 `maxs[i] > mins[i]`；射线方向非零；`material` 仅接受 `None`
  或非空 `str`（拒绝 `bool`、数字、空串、bytes 等）。
* 非法输入抛出 `GeometryError`，消息含可定位路径（如
  `Sphere(leaf_id='s1').center[2]: ...`、`tree.children[0].positive: ...`），
  不做任何静默纠正。`ray_intervals` 为纯函数：输入校验失败时不产生任
  何部分结果或副作用。
* `CertificationBudgetExhausted`（`CSGError` 子类）与"证明无交"严格分
  开：无交返回空结果；只有当两个**不同**事件参数接近到 240 位 Decimal
  预算内仍无法证明其先后顺序时才抛出该异常（小整数系数下实际不会触
  发）。

## 浮点误差口径

* 所有事件参数在内部精确表示：盒事件、有理球根、切点均为
  `Fraction`；球的非有理根为精确二次根对象（Fraction 系数 + 根号符
  号）。
* 重合判定（同 t 聚类）是**精确**的：Fraction 直接比较；根与有理数由
  二次式符号精确判定；根与根先精确判等（同最小多项式），不同根的次
  序用 60→120→240 位 Decimal 升级认证，超预算则抛
  `CertificationBudgetExhausted`，绝不靠宽容差猜测。
* 仅在输出边界做一次舍入：`t` 经 `float()` 转换，有理端点误差 ≤ 1
  ulp；无理端点先取 80 位 Decimal 再转 float，相对误差 ≲ 1e-15。
* 法向：盒面法向精确；球法向在命中点精确计算（有理 t）或经 70 位
  Decimal（无理 t），相对误差 ≲ 1e-15。
* 朝向 `orientation` 是**精确**的：分类只看外法向与方向点积的符号，盒面
  为整数符号、球为 Fraction 符号（无理横穿面按根序判定），不含任何浮点
  舍入；grazing 即点积严格等于 0。
* 因此边界之间不存在"宽容差"：区间不相交、不重叠是由精确排序保证的，
  不是靠 epsilon 合并。

## 设计取舍

* **三值状态扫描**：每叶在每段开区间上取 IN / T（零测度接触）/ OUT 三
  态，布尔运算按三态真值表求值（减体为闭集），再对 T 段做正则化归
  并。这样切线、贴面、共点等退化情形与常规情形走同一条代码路径。
* **精确算术优先**：小整数系数下全部比较都是精确的 Fraction 运算，
  Decimal 仅用于无理根次序的认证升级，兼顾正确性与性能（100 节点规
  模单次求交为毫秒级）。
* **来源代表与全集**：`sources` 保留全部来源面，代表按叶 ID 字典序确
  定性地选出，便于确定性下游处理。

## 已知限制

* 叶仅球与轴对齐盒（无旋转盒、无其他图元、无变换节点）；布尔算子仅
  union / intersection / difference。这是本题的数学子集，不宣称完整
  CAD 内核或行业标准兼容。
* 树为真树（不允许节点共享）；节点数上限 100 为接口边界。
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
  _types.py         # 公开数据结构、输入校验、错误类型、朝向常量
  _exact.py         # Fraction/二次根精确比较与 Decimal 升级认证
  _core.py          # 叶事件生成、三值扫描、区间/边界/接触装配、材料与朝向
tests/
  test_analytic.py  # 解析精确端点（同心空心球）、重合盒面、切线、内部起射、错误语义
  test_sampling.py  # 独立参考分类器的小采样成员测试（不调用被测核心）
  test_materials.py # 材料 ID、front/back/grazing 朝向、嵌套差奇偶、相切/重合端点
demo.py             # 固定输入演示：材料/朝向 + 嵌套差 + 相切边界 + 输入拒绝
```
