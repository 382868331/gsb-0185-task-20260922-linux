# csg_ray — CSG 射线区间求交（带空腔）

离线 CAD 几何用的可复用小模块：把一条射线与一棵 CSG（构造实体几何）树做
1D 区间分类，输出实体参数区间、零测度切线接触（touch）和最近的正向实体
边界，并给出每个边界的全部来源面。仅依赖 Python 标准库。

## 运行环境

- Python **3.14.7**（仅使用 3.10+ 即有的语法/标准库特性）
- Linux 原生运行，**无需安装任何第三方依赖**，无需网络/账号/密钥/数据库/
  Docker/硬件
- 核心、测试、演示全部可离线执行

## 准确命令

```bash
# 必要测试（40 个用例）
python -m unittest discover -s tests -v

# 固定输入演示：真实计算一个同心空心球（展示空腔出入面）
# 并真实触发一次输入拒绝
python demo.py
```

演示无 `sleep`、无预录结果，正常在亚秒级完成（远低于约 8 秒上限）。

## 公开接口与数据结构

`from csg_ray import ...`

| 名称 | 说明 |
| --- | --- |
| `ray_intervals(tree, ray) -> RayIntervalResult` | 主入口，纯函数 |
| `solid_at(result, t) -> bool` | 闭区间成员判定（touch 不算内部） |
| `MAX_NODES` | 树节点数上限，常量 `100` |

树节点（均为 frozen dataclass）：

```python
Sphere(leaf_id: str, center: (x,y,z), radius: float > 0)
Box(leaf_id: str, min_corner: (x,y,z), max_corner: (x,y,z))  # 各轴严格小于
Union(left, right)
Intersection(left, right)
Difference(left, right)        # left \\ right，right 即"被减体/空腔"
Ray(origin: (x,y,z), direction: (dx,dy,dz))  # direction 非零
```

返回：

```python
RayIntervalResult(
    intervals=(Interval(t_enter, t_exit, enter_sources, exit_sources), ...),
    touches=(Touch(t, sources), ...),
    nearest_boundary=Boundary(t, "enter"|"exit", sources) | None,
)
FaceSource(leaf_id, face, normal, flipped)
```

- `intervals` 为参数 `t` 上**排序、互不相交的闭区间**（`p(t)=origin+t*dir`，
  `t` 可取负数，射线是整条有向直线；区间端点表示实体的进出位置）。
- `touches` 为零测度相切点：球体切线接触、盒的边/角单点接触；**不计入
  实体内部**，也不产生 `nearest_boundary`。
- `nearest_boundary` 为满足 **`t > 0`（严格正向）** 的最近实体边界；射线
  原点表面 `t=0` 不计为正向。几何完全未命中时为 `None`，且**不是错误**。
- 面名：球为 `"surface"`；盒为 `"x-" "x+" "y-" "y+" "z-" "z+"`。`normal`
  是该面在**结果实体**上的有效单位外法向；被减体（`Difference` 的 right
  子树）面法向整体翻转，`flipped=True` —— 这些就是空腔壁。嵌套时按
  "被减次数"奇偶决定（差集右子树中的差集右子树翻转抵消）。
- 同一边界有多个可用来源时，来源按 **(leaf_id 字典序, 面名)** 排序后全部
  保留；`Interval.representative_enter/exit`、`Touch.representative`、
  `Boundary.representative` 取字典序最小的代表。

## 输入上限与错误语义

- 树至多 `MAX_NODES = 100` 个节点；叶子 `leaf_id` 非空且全树唯一；输入必须
  是树（节点对象不可被两个父节点共享，拒绝 DAG）。
- 所有数值输入拒绝 `NaN`/`±Infinity`（抛 `InvalidRayError`/
  `InvalidGeometryError`）；`bool` 是 `int` 子类，数值位一律拒绝（`TypeError`）。
  非法输入抛出**带定位路径**的错误（标明节点与字段，如
  `tree.Union.left/Sphere('s').radius`），不做静默纠正。
- 几何上的"无解"（射线完全错过实体）返回空结果，与资源耗尽严格区分：
  超过 100 节点抛 `BudgetExceededError`。
- `ray_intervals` 是纯函数：失败时在返回前抛出，不存在"部分变更"状态。

异常类型：`CSGError(ValueError)` 为基类，下分 `InvalidGeometryError`、
`InvalidRayError`；`BudgetExceededError(RuntimeError)`；类型错误用内置
`TypeError`。

## 算法与设计取舍

1. **每叶生成事件**：球求二次方程；盒做三轴 slab 求交，近/远平面分别给出
   enter/exit 事件，边（两面）与角（三面）重合时合并为单个 touch。
2. **精确有理数 + 必要时 Decimal 提升**：输入先转 `fractions.Fraction`
   （对二进制浮点是精确转换），球判别式的正负/重根判定优先用精确有理数；
   仅当浮点判别式落在零附近的相对带（`1e-12`）内且需高精度开方时，用
   `Decimal`（60 位）重算根，避免把近重根误判成相切。
3. **同 t 事件成批处理**：同一参数值的全部事件合为一组，自底向上求每个
   CSG 节点的"组前是否在内 / 组后是否在内 / 该点是否属于集合"三态。只有
   根集合状态真正翻转才产生边界；**布尔操作下状态不变的重合事件不产生
   伪实体边界**（例如两盒共面 union 时 `x=2` 处无边界、共面 difference
   无幻影面）。
4. 区间端点先以 `Fraction` 全程比较与排序，仅在输出时转 `float`。
5. 面来源通过 (t, leaf_id, face) 反查事件携带的精确法向；被减体法向在
   输出阶段统一翻转。

## 浮点误差口径（不用宽容差掩盖边界错误）

- **拓扑/端点判定不含容差**：区间有无、相切 vs 相交、重合事件分组全部基于
  精确 `Fraction`/`Decimal` 比较；输入是小整数/短小数（可精确表示为
  Fraction 的二进制浮点）时，端点在有理数意义下精确，测试中以 `==` 核对。
- 输出的 `t` 是精确有理值转 `float` 的**一次舍入**（≤ 0.5 ULP 相对误差）；
  球法向同理。非二进制可精确表示的一般 `float` 输入也按其二进制位的精确
  值处理，因此不存在"输入误差"被放大或被容差吞掉的情况。
- 测试中的采样核对只比较区间**内部**成员（严格内点，避开边界），边界本身
  由解析同心空心球用精确等式核对，不以宽容差放行。
- 限制：`float` 输出无法表达两个比 ~1e-16 相对差更近的不同端点（内部仍
  正确区分，输出舍入后可能相等）；这是 IEEE-754 双精度的表示限制。

## 已知限制（本契约的数学子集，非行业标准完整兼容）

- 叶仅支持球与轴对齐盒；无一般多边形/曲面、无变换节点。
- CSG 按本模块定义的闭/开约定处理：实体区间为闭区间，但切线接触（零测度）
  单列 touch 而不算实体内部。
- 射线方向某分量为 0 且原点**恰好位于盒的某个面平面内**（射线贴着面走）
  时，该 2D 面运行段不产生区间（面不是体内部）；此退化情形按"非实体"
  处理。边/角的单点接触仍正常给出 touch。
- 被减体的**切点严格落在被保留实体内部**时，会在结果上形成一个零测度
  "针孔"点；闭区间无法表达单点镂空，该点并入相邻实体区间，不另出标记
  （切点位于实体外部时的正常 touch/抑制行为均已覆盖测试）。
- `nearest_boundary` 只报告实体区间边界（enter/exit），不报告 touch 点。
- 不做包围盒加速；规模上限 100 节点，面向正确性而非渲染级性能。
- 不宣称兼容 STEP/OpenCSG 等任何行业标准。

## 文件

- `csg_ray.py` — 核心库（唯一库模块）
- `tests/test_csg_ray.py` — 必要测试；期望值来自独立的严格点包含参考实现
  （不调用被测核心）与解析同心空心球
- `demo.py` — 固定输入演示
