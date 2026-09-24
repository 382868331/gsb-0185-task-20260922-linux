# csg_rays — 带空腔的 CSG 射线区间求交（表面材料与朝向）

离线 CAD 几何工程用可复用模块：对 CSG 树（叶仅球与轴对齐盒，内部节点
union / intersection / difference）与一条射线求交，输出排序不相交的实体
t 区间、零测度相切接触、带进入/退出面来源（含**材料 ID 与创作朝向**）的
实体边界，以及最近正向实体边界。

* **Python 版本**：3.14.7（仅用标准库：`fractions` / `decimal` / `math` /
  `dataclasses` / `functools` / `types`）。
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
    MAX_TREE_NODES, DEFAULT_MATERIAL, BOX_FACES,
)

tree = Difference(
    Sphere((0, 0, 0), 5, "outer", material="shell"),
    Sphere((0, 0, 0), 2, "inner", material="liner"),
)
res = ray_intervals(tree, Ray((-10, 0, 0), (1, 0, 0)))
# res.intervals -> (Interval(5.0, 8.0, ...), Interval(12.0, 15.0, ...))
# res.nearest_boundary -> Boundary(t=5.0, kind="enter", ...)
# 空腔内壁的 source.material == "liner"、orientation == "inward"、flipped is True
```

### 树节点

| 构造 | 说明 |
| --- | --- |
| `Sphere(center, radius, leaf_id, material=DEFAULT_MATERIAL, face_materials=None)` | 实体球叶。`radius > 0`；球面材料由 `material` 给定，`face_materials` 仅接受键 `"sphere"`。 |
| `Box(mins, maxs, leaf_id, material=DEFAULT_MATERIAL, face_materials=None)` | 轴对齐实体盒叶。每轴 `maxs[i] > mins[i]`；六个面默认用 `material`，可用 `face_materials={"x_lo": ..., ...}` 逐面覆盖。 |
| `Union(children)` / `Intersection(children)` | n 元并 / 交，`children` 非空可迭代。 |
| `Difference(positive, *subtractors)` | 正体减去一个或多个减体（正则化差，减体按闭实体处理）。 |
| `Ray(origin, direction)` | 射线 `origin + t*direction`，`t ∈ ℝ`（整条直线）；`direction` 必须非零。 |

每个叶节点暴露：

* `material`：构造时给出的原语级材料 ID（校验后的非空 `str`，默认
  `"default"`，常量 `DEFAULT_MATERIAL`）。
* `materials`：只读映射 `{面名: 材料ID}`，球为 `{"sphere": ...}`，盒为六个
  面名（元组 `BOX_FACES`）到各自材料 ID 的解析结果；**来源上报的
  `material` 永远取自这里命中的那个面**。
* `face_materials`：规范化后的逐面覆盖（无覆盖为 `None`，否则为按面名排序
  的 `(面, 材料)` 元组——用值类型元组而非可变 dict，frozen 节点保持可比较、
  可哈希）。

材料 ID 必须是非空 `str`（拒绝 `bool`/空串/非字符串）；`face_materials`
必须是面名到材料 ID 的映射，**未知面名直接报错**（不静默忽略，避免留下
半配置表面）。位置参数 `center/radius/leaf_id` 等与旧版完全一致，旧的
三叶参数调用无需改动。

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
  按 `(leaf_id, face)` 字典序排序；`source` 为代表，即排序首位（叶 ID 优先，
  叶 ID 相同再按面名）。材料名不参与代表选择。
* `Source` 字段：
  * `leaf_id`、`face`（球为 `"sphere"`；盒为
    `"x_lo"/"x_hi"/"y_lo"/"y_hi"/"z_lo"/"z_hi"`）、`kind`
    （叶侧事件 `"enter"/"exit"/"touch"`）。
  * `material`：该命中面的材料 ID，取自叶节点解析后的 `materials[face]`；
    并/交/差只做来源合成，**材料 ID 原样透传，不做合并或改名**。同一边界
    多来源时各自保留自己的材料。
  * `orientation`：表面的**创作朝向**，布尔合成中不变。正体叶为
    `"outward"`；处于奇数个 `Difference` 减体位置的面为 `"inward"`（差集
    空腔内壁）；嵌套差集按奇偶翻转（内层差集的正体翻回 `"outward"`，其
    减体再翻为 `"inward"`）。
  * `normal`：命中点处单位外法向（3 元 float 元组）。
  * `flipped`：报告的几何法向是否相对创作外法向取反。规则与
    `orientation` 区分如下：
    * 实体**边界**上，减体内壁翻转法向，`flipped=True`，与
      `orientation=="inward"` 一致（指向空腔内部）；
    * **零测度接触（touch）不是任何空腔的壁**，减体面即使创作朝向为
      `"inward"` 也报告基本体真实外法向，`flipped=False`。因此在相切接触处
      可能出现 `orientation=="inward"` 而 `flipped is False`：前者记录 CSG
      角色，后者描述几何法向。

## 语义约定

* **正则化闭实体**：实体按闭集理解，输出内部为开区间。切线接触、贴面
  接触等零测度集合不产生实体内部：被实体夹住的相切事件被"吞并"（不
  在实体上戳出零宽洞），两侧皆空的相切事件报告为 `Touch`。
* **同一 t 的事件批量处理**：重合事件按同一 t 聚类后一次性更新状态；
  不引起状态改变的重合事件**不**生成伪实体边界（如两盒共面拼接的并
  只产生一个区间；完全相同两叶的差结果为空且无边界）。
* **重合端点的代表优先级**：多个来源在同一 t 共同构成边界/接触时，全部
  来源都保留，代表固定取 `(leaf_id, face)` 字典序首位，确定性且与材料名、
  法向无关。
* **材料与朝向的合成**：并/交取实际造成状态切换的叶的面及其材料；差集
  记录正体进入面与减体退出面（内壁），嵌套差集朝向按差集嵌套奇偶累计。
* **来源判定**：叶在某边界处有事件，且强制该叶为 IN / OUT 会改变结果
  状态时，该叶面才计入 `sources`；接触点（点态 T）的来源在**点簇**上做
  IN/OUT 反事实判定（贴面/相切接触的因果在点态而非开段上）。因此无影响
  的重合事件不会出现在来源列表中，而相切减体制造的接触能正确归属。
* **射线是整条直线**：区间覆盖所有 `t ∈ ℝ`；`nearest_boundary` 额外筛
  选 `t > 0`（"最近正向"）。从实体/空腔内部起射时，负 t 区间同样给出。

## 输入上限与错误语义

* 树节点总数（叶 + 内部节点）≤ `MAX_TREE_NODES = 100`；树不得共享节点
  （同一对象出现两次即报错）；`leaf_id` 全局唯一。
* 数值输入拒绝 `bool`、NaN、Infinity；向量必须恰为 3 个数值；球半径
  `> 0`；盒 `maxs[i] > mins[i]`；射线方向非零。
* 材料 ID（`material` 与 `face_materials` 的值）必须是非空 `str`；
  `face_materials` 必须是映射，键必须是该叶合法面名（球仅 `"sphere"`，
  盒为六面名之一），值必须是合法材料 ID。未知键 / 非字符串值 / 非映射
  均在叶构造时抛 `GeometryError`。
* 非法输入抛出 `GeometryError`，消息含可定位路径（如
  `Sphere(leaf_id='s1').center[2]: ...`、
  `Box(leaf_id='b1').face_materials['x_lo']: ...`、
  `tree.children[0].positive: ...`），
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
* 因此边界之间不存在"宽容差"：区间不相交、不重叠是由精确排序保证的，
  不是靠 epsilon 合并。

## 设计取舍

* **三值状态扫描**：每叶在每段开区间上取 IN / T（零测度接触）/ OUT 三
  态，布尔运算按三态真值表求值（减体为闭集），再对 T 段做正则化归
  并。这样切线、贴面、共点等退化情形与常规情形走同一条代码路径。
* **精确算术优先**：小整数系数下全部比较都是精确的 Fraction 运算，
  Decimal 仅用于无理根次序的认证升级，兼顾正确性与性能（100 节点规
  模单次求交为毫秒级）。
* **来源代表与全集**：`sources` 保留全部来源面，代表按 `(leaf_id, face)`
  字典序确定性地选出（材料名不参与），便于确定性下游处理。
* **材料原样透传、朝向与法向分离**：布尔算子不改写材料 ID，命中哪个面
  就带哪个面的材料；`orientation` 记录表面在 CSG 树中的创作角色（差集
  嵌套奇偶，不可变），`flipped`/`normal` 描述该几何点上实际报告的法向，
  使"相切处的减体面"既能标注内朝向角色、又不被错误地当作空腔内壁翻转。
* **接触来源在点簇上反事实判定**：贴面/相切接触是点态（T）现象，段态
  反事实会漏掉"正体为 IN、减体相切制造接触"这类因果，旧实现在该路径上
  会得到空来源列表（极端场景内部错误）。本次将接触来源改为点簇 IN/OUT
  反事实，边界来源仍走段态判定。

## 已知限制

* 叶仅球与轴对齐盒（无旋转盒、无其他图元、无变换节点）；布尔算子仅
  union / intersection / difference。这是本题的数学子集，不宣称完整
  CAD 内核或行业标准兼容。
* 材料只是与基本体表面绑定的不透明 ID：本模块不解释、不合并材料，也不
  提供材料属性表或纹理/着色；同一边界多来源并列时，材料如何取舍由下游
  决定。
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
  _types.py         # 公开数据结构、材料/输入校验、错误类型
  _exact.py         # Fraction/二次根精确比较与 Decimal 升级认证
  _core.py          # 叶事件生成、三值扫描、区间/边界/接触/材料装配
tests/
  test_analytic.py    # 解析精确端点（同心空心球）、重合盒面、切线、内部起射、错误语义
  test_sampling.py    # 独立参考分类器的小采样成员测试（不调用被测核心）
  test_materials.py   # 材料ID/朝向传播、嵌套差集奇偶、相切边界、代表优先级、材料校验
demo.py               # 固定输入演示：嵌套差集朝向、逐面材料、共面合并、相切减体、输入拒绝
```
