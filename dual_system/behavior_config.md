# BEHAVIOR-1K 配置参考手册

本文档提供 RLinf 中 BEHAVIOR-1K 任务和场景初始化配置的完整参考。

## 支持的任务（0-49）

BEHAVIOR-1K 包含 50 个家务操控任务。要切换任务，请在配置 YAML 中设置 `omni_config.task.activity_name`。

| 编号 | `activity_name` | 说明 |
|------|-----------------|------|
| 0 | `turning_on_radio` | Turn on the radio receiver that's on the table in the living room. |
| 1 | `picking_up_trash` | Put the three can of soda from the living room inside the tash can in the kitchen. |
| 2 | `putting_away_Halloween_decorations` | Place each of the two pumpkins and all three candles from the living room inside a cabinet in the living room (use any cabinet), then make sure every cabinet is closed, and position the cauldron so it is next to a table in the living room. |
| 3 | `cleaning_up_plates_and_food` | From the breakfast table in the kitchen, move both pizzas - keeping each on its plate - into the same refrigerator, put both bowls into one sink, and make sure the refrigerator is closed. |
| 4 | `can_meat` | Open the kitchen cabinet, take out the two hinged jars, open them, place exactly two cooked bratwursts from the chopping board on the countertop into each jar, then close both jars, put them back inside the cabinet, and close the cabinet. |
| 5 | `setting_mousetraps` | Take the four mousetraps from the cabinet in the bathroom and place them on the bathroom floor. Make sure all four end up on the same floor surface, and ensure that at least two of them are either under or directly next to the same bathroom sink. |
| 6 | `hiding_Easter_eggs` | Take the three Easter eggs out of the wicker basket on the lawn in the garden, then place them on the lawn next to a single tree (choose any tree) so that all three eggs are next to the same tree and none are left in the basket. |
| 7 | `picking_up_toys` | Put all the toys in the child's room - the three board games (two on the bed and one on the table), the two jigsaw puzzles on the table, and the tennis ball on the table - inside the toy box on the table in the child's room. |
| 8 | `rearranging_kitchen_furniture` | Move the toaster, food processor, and French press from the kitchen countertop into the same kitchen cabinet, and make sure that cabinet is closed at the end. |
| 9 | `putting_up_Christmas_decorations_inside` | In the living room, take the wreath, three candy canes, and two pillar candles out of the wicker basket. Place the wreath and two of the candy canes on the same living-room sofa. Put the remaining candy cane on top of a dining-room table. Put both pillar candles together on top of one dining-room table (they can share the same table). Finally, place all three gift boxes under or right next to the Christmas tree in the living room. |
| 10 | `set_up_a_coffee_station_in_your_kitchen` | Set up a coffee station on the kitchen countertop: keep the coffee maker on the countertop, move the bottle of coffee from the kitchen shelf to the counter next to the coffee maker, place a paper coffee filter on top of the coffee maker, put the saucer next to the coffee maker with the coffee cup on the saucer, and place the electric kettle next to the coffee maker. |
| 11 | `putting_dishes_away_after_cleaning` | In the kitchen, gather all eight plates from the two countertops, place them all inside a single cabinet (either one), and make sure all cabinets are closed when you're done. |
| 12 | `preparing_lunch_box` | Put both apple halves, the club sandwich, and the chocolate chip cookie from the chopping board on the kitchen countertop into the packing box on the countertop. Then take the bottle of tea out of the refrigerator, put it into the same box, and close the refrigerator when you're done. |
| 13 | `loading_the_car` | Put the digital camera from the living room table into the container on the living room floor. Then take the container and the tennis racket to the garage, place both in the car trunk, and close it. |
| 14 | `carrying_in_groceries` | Take the sack of groceries out of the car trunk in the garage, bring it to the kitchen, and put both the tomato and the carton of milk into the refrigerator in the kitchen. When you're done, close the car trunk and make sure the refrigerator in the kitchen is closed. |
| 15 | `bringing_in_wood` | Bring the three plywood sheets from the garden into the corridor and place them on the floor there. |
| 16 | `moving_boxes_to_storage` | Move the two storage containers from the living room to the garage. In the garage, place one container on the floor and stack the other container on top of it (either order is fine). |
| 17 | `bringing_water` | Retrieve the two bottles from the refrigerator in the kitchen, bring them to the living room, and place both on the coffee table. Make sure the refrigerator is closed when you finish. |
| 18 | `tidying_bedroom` | In the bedroom, move the book from the bed onto either nightstand, and place the two sandals side by side next to the bed. |
| 19 | `outfit_a_basic_toolbox` | In the utility room, put the drill, pliers, flashlight, Allen wrench, and screwdriver from the tabletop into the toolbox, keep the toolbox on the tabletop, and close the toolbox. |
| 20 | `sorting_vegetables` | Sort the vegetables from the two wicker baskets on the kitchen floor into the mixing bowls on the kitchen countertop: put all three bok choy and all three Vidalia onions together into one mixing bowl; put both leeks and both broccoli together into a second mixing bowl; and put all three sweet corn into a third mixing bowl. |
| 21 | `collecting_childrens_toys` | Pick up the two dice from the bed, the two teddy bears from the floor, and the two board games (one from the desk and one from the bed), and place them all inside the same bookcase in the child's room. |
| 22 | `putting_shoes_on_rack` | Pick up the two gym shoes and the two sandals from the corridor floor and place them onto the hallstand (shoe rack) in the corridor, making sure they are on the rack and not on the floor. Arrange them so the two gym shoes are next to each other and the two sandals are next to each other. |
| 23 | `boxing_books_up_for_storage` | Put all six books from the bookcases in the living room into the box on the living room floor. |
| 24 | `storing_food` | Put away all the food on the kitchen countertop by storing it inside the kitchen cabinets: move the two boxes of oatmeal, two bags of chips, two bottles of olive oil, and two jars of sugar from the countertop into the kitchen cabinets (each item can go into either cabinet). |
| 25 | `clearing_food_from_table_into_fridge` | Pack the half chicken and the half apple pie from the plates on the breakfast table into the two tupperware containers from the countertop, then put both tupperware containers inside the refrigerator in the kitchen and make sure the refrigerator is closed at the end. |
| 26 | `assembling_gift_baskets` | Place one candle, one butter cookie, one piece of Swiss cheese, and one bow from the table into each of the four wicker baskets on the floor in the living room. |
| 27 | `sorting_household_items` | From the two baskets on the bedroom floor, take out the items and organize them in the bathroom: place both detergent bottles under the bathroom sink next to each other; put the box of sanitary napkins on the bathroom shelf; set the soap dispenser on the sink; make sure the cup remains on the sink; put both the toothpaste tube and the toothbrush inside the cup. |
| 28 | `getting_organized_for_work` | In the bedroom, organize the workspace by keeping the computer under the desk, ensuring the monitor is on the desk, placing the keyboard on the desk next to the monitor, placing the mouse on the desk next to the keyboard, moving the folder from the swivel chair onto the desk next to the mouse, stacking the notebook on top of the folder with the pen on top of the notebook, and positioning the swivel chair next to the desk. |
| 29 | `clean_up_your_desk` | In the child's room, clean up the desk: put both folders and both paperback books into the bookcase; put the pencil and both pens into the pencil case and leave the case on the desk; take the stapler out of the bookcase and place it on the desk; move the laptop from the bed onto the desk and close it. |
| 30 | `setting_the_fire` | In the living room, place the newspaper from the table into the wood fireplace, then put both pieces of firewood from the floor on top of the newspaper. Use the cigar lighter to ignite any one of the items so that the whole pile catches fire, and then turn the lighter off. |
| 31 | `clean_boxing_gloves` | Wash the two dusty boxing gloves from the countertop in the utility room in the washer until they are no longer covered with dust. |
| 32 | `wash_a_baseball_cap` | Wash the two baseball caps on the countertop in the utility room using the washer until they are no longer dirty. |
| 33 | `wash_dog_toys` | In the utility room, take the two teddy toys, the tennis ball, and the softball out of the cabinet and wash them in the washer so that both teddies are free of dirt and dust, the tennis ball has no debris, and the softball has no dirt. |
| 34 | `hanging_pictures` | Pick up the poster from the kitchen countertop and hang it on one of the wall nails in the kitchen. |
| 35 | `attach_a_camera_to_a_tripod` | Attach the digital camera to the camera tripod in the bedroom. |
| 36 | `clean_a_patio` | Pick up the broom in the garden and sweep the mud off the patio floor until the floor is no longer covered in mud. |
| 37 | `clean_a_trumpet` | In the bedroom, pick up the scrub brush from the desk and scrub the cornet (trumpet) on the desk until it's no longer covered in dust. |
| 38 | `spraying_for_bugs` | Pick up the pesticide atomizer in the garden and spray insectifuge to fully cover both potted plants in the garden. |
| 39 | `spraying_fruit_trees` | In the garden, pick up the pesticide atomizer on the floor and spray pesticide onto both trees until each tree trunk is fully covered. |
| 40 | `make_microwave_popcorn` | In the kitchen, take the popcorn bag from the countertop, put it into the microwave, and heat it until the popcorn is cooked so the cooked popcorn ends up inside the bag. |
| 41 | `cook_cabbage` | From the kitchen refrigerator, take the cabbage and the chili, dice them on the chopping board with the knife, cook the diced cabbage and diced chili in the frying pan on the stove, and leave the cooked, diced cabbage and cooked, diced chili in the frying pan. |
| 42 | `chop_an_onion` | In the kitchen, take the Vidalia onion out of the sink, dice it on the chopping board with the paring knife, put the diced onion into the bowl on the countertop, then place both the paring knife and the chopping board into the sink. |
| 43 | `slicing_vegetables` | From the refrigerator in the kitchen, take out the two bell peppers, the two beets, and the zucchini. Then, on either chopping board on the countertop, use the parer to dice all of them so that only diced bell pepper, diced beet, and diced zucchini remain. Make sure the refrigerator is closed when you finish. |
| 44 | `chopping_wood` | Chop the four logs on the driveway in the garden into eight half logs using the axe and the chopping block. |
| 45 | `cook_hot_dogs` | Take the two hot dogs out of the refrigerator in the kitchen and cook them in the microwave until both are cooked. |
| 46 | `cook_bacon` | Take the tray with six slices of bacon out of the refrigerator in the kitchen, cook all six slices in the frying pan on the stove until they're cooked, and make sure the refrigerator is closed when you're done. |
| 47 | `freeze_pies` | In the kitchen, take the two apple pies from the plates on the countertop, put each pie into a separate tupperware container taken from the cabinet, place both tupperwares inside the refrigerator, close the refrigerator, and leave them until the pies are frozen. |
| 48 | `canning_food` | In the kitchen, open the refrigerator and the cabinet. Take the steak and the pineapple out of the refrigerator and take two bowls from the cabinet. On the chopping board on the countertop, use the carving knife to dice the steak and to dice the pineapple. Put only the diced steak into one bowl and only the diced pineapple into the other bowl - do not mix them. Place both bowls back inside the cabinet, then close the refrigerator and close the cabinet. |
| 49 | `make_pizza` | Make a pizza on the cookie sheet in the kitchen: take the grated cheese, the four pieces of pepperoni, and the two mushrooms from their tupperware containers in the refrigerator; chop the Vidalia onion on the chopping board with the knife, and also chop the two whole mushrooms in half; top the pizza dough that's already on the cookie sheet with the cheese, pepperoni, halved mushrooms, and chopped onion; bake it in the oven until it becomes a pizza, and leave the finished pizza on the cookie sheet. |

## 场景初始化模式

`omni_config.task.instance_resample_mode` 设置控制每个 episode 开始时场景的初始化方式。

### 模式 1：`disabled`（默认） -- 固定实例

```yaml
omni_config:
  task:
    instance_resample_mode: disabled
    activity_instance_id: 0
    activity_instance_dir: null
    online_object_sampling: False
    use_presampled_robot_pose: True
```

**行为**：每个 episode 重置为完全相同的场景状态。
- 若 `activity_instance_dir` 为 `null`，使用 OmniGibson 内置的默认实例。
- 若 `activity_instance_dir` 已设置，从该目录加载 id=`activity_instance_id` 的实例。

#### 在 `disabled` 模式下使用特定 `tro_state` 实例

当 `activity_instance_dir` 指向一个 `tro_state` 文件目录，且需要评估特定实例（例如 instance 242）时，还必须将 `scene.scene_instance` 设置为已存在的 `_template.json` 名称。这是因为 OmniGibson 在初始化时需要完整的 template 文件来构建场景，而 `tro_state` 文件仅在场景加载后替换任务相关的物体状态。

```yaml
omni_config:
  task:
    activity_name: turning_on_radio
    activity_instance_id: 242                     # 重置时加载的目标 tro_state 实例
    instance_resample_mode: disabled
    activity_instance_dir: /path/to/2025-challenge-task-instances/scenes/house_double_floor_lower/json/house_double_floor_lower_task_turning_on_radio_instances/
    instance_file_format: tro_state
    online_object_sampling: False
    use_presampled_robot_pose: True
  scene:
    type: RLinfInteractiveTraversableScene
    scene_model: house_double_floor_lower
    scene_instance: house_double_floor_lower_task_turning_on_radio_0_0_template  # 初始化用的完整 template
```

**执行过程**：
1. OmniGibson 使用 `scene_instance` 指定的 template（`*_0_0_template.json`）初始化场景。
2. RLinf 的 `ActivityInstanceLoader` 扫描 `activity_instance_dir` 中的 `tro_state` 文件。
3. 每次重置时，加载器应用 `activity_instance_id=242` 对应的 `tro_state` 文件，就地替换任务相关物体的位置和机器人位姿。

**为什么这里需要 `scene_instance`，而默认 `activity_instance_id: 0` 时不需要**：

当配置中未设置 `scene_instance` 时，OmniGibson 的 `BehaviorTask.verify_scene_and_task_config()` 会从 `activity_instance_id` 自动生成：

```
scene_instance = "{scene_model}_task_{activity}_{definition_id}_{instance_id}_template"
```

使用默认 `activity_instance_id: 0` 时，会生成 `house_double_floor_lower_task_turning_on_radio_0_0_template`，其对应的 `_template.json` 存在于 `task_dir/json/` 目录中，因此无需显式设置 `scene_instance`。

当 `activity_instance_id` 改为 242 时，自动生成的名称变为 `*_0_242_template`，但官方数据集仅为非零 instance ID 提供 `tro_state` 文件（没有 `_template.json`）。显式设置 `scene_instance` 可以覆盖自动生成逻辑，使 OmniGibson 加载已有的 id=0 template。

### 模式 2：`offline` -- 从缓存中随机采样

```yaml
omni_config:
  task:
    instance_resample_mode: offline
    activity_instance_dir: /path/to/cached_instances/  # 必填
    instance_file_format: tro_state  # 或 "template"
    online_object_sampling: False
    use_presampled_robot_pose: True
```

**行为**：启动时 RLinf 扫描 `activity_instance_dir`。每次重置前，随机采样一个实例。

支持两种格式：
- **`tro_state`**：轻量级，就地恢复任务相关物体 + 机器人位姿
- **`template`**：完整场景重载，彻底重置但速度较慢

#### 在 `offline` 模式下使用 `tro_state` 文件

在 `offline` 模式下使用 `tro_state` 格式时，同样必须设置 `scene.scene_instance`，原因与 `disabled` 模式相同：OmniGibson 需要完整的 template 来初始化场景，`tro_state` 文件只能在初始化之后替换物体状态。

```yaml
omni_config:
  task:
    activity_name: turning_on_radio
    instance_resample_mode: offline
    activity_instance_dir: /path/to/2025-challenge-task-instances/scenes/house_double_floor_lower/json/house_double_floor_lower_task_turning_on_radio_instances/
    instance_file_format: tro_state
    online_object_sampling: False
    use_presampled_robot_pose: True
  scene:
    type: RLinfInteractiveTraversableScene
    scene_model: house_double_floor_lower
    scene_instance: house_double_floor_lower_task_turning_on_radio_0_0_template
```

**执行过程**：
1. OmniGibson 使用 `scene_instance` 指定的 template 初始化场景。
2. RLinf 的 `ActivityInstanceLoader` 扫描 `activity_instance_dir` 中的所有 `tro_state` 文件。
3. 每次重置时，通过 `random.choice()` 选取一个实例，并就地应用其 tro_state，替换任务相关物体位置和机器人位姿。

注意：`offline` 模式下 `activity_instance_id` 被忽略（实例始终随机选取）。若 `instance_file_format` 为 `template` 而非 `tro_state`，则不需要设置 `scene_instance`，因为每个 template 文件包含完整的场景定义。

### 模式 3：`online` -- 实时 BDDL 采样

```yaml
omni_config:
  task:
    instance_resample_mode: online
    online_object_sampling: True
    use_presampled_robot_pose: False
    activity_instance_dir: null  # 不可设置
```

**行为**：每次重置前，OmniGibson 的 BDDL 约束求解器生成一个新场景。多样性最高但速度最慢。

## 下载与生成初始化样本

### 官方挑战赛实例

```bash
export OMNIGIBSON_DATA_PATH=/path/to/BEHAVIOR-1K-datasets
python -c "from omnigibson.utils.asset_utils import download_2025_challenge_task_instances; download_2025_challenge_task_instances()"
```

下载到 `$OMNIGIBSON_DATA_PATH/og_dataset/2025-challenge-task-instances/<scene_model>/json/<scene_model>_task_<activity_name>_instances/`

### RLinf 实例生成器

```bash
cd /path/to/RLinf

# 生成 tro_state 实例（推荐）
python rlinf/envs/behavior/instance_generator.py \
  --config examples/embodiment/config/env/behavior_r1pro.yaml \
  --output-format tro_state \
  --start-idx 1 \
  --end-idx 50

# 生成完整 template 实例
python rlinf/envs/behavior/instance_generator.py \
  --config examples/embodiment/config/env/behavior_r1pro.yaml \
  --output-format template \
  --start-idx 1 \
  --end-idx 50
```

## 配置参考

### 顶层 RLinf 字段

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `env_type` | `behavior` | 环境类型标识符。 |
| `auto_reset` | `False` | 在 epoch 内自动重置已完成的环境。仅在 eval-only 且 `ignore_terminations=True` 时设为 `True`。 |
| `ignore_terminations` | `False` | 将所有 episode 结束视为截断（不产生任务完成信号）。 |
| `use_rel_reward` | `True` | 使用相对/基于势函数的奖励。 |
| `max_episode_steps` | `2000` | Episode 硬限制（RLinf 在精确步数处强制截断；OmniGibson 存在 off-by-one 问题）。 |
| `max_steps_per_rollout_epoch` | `2000` | 每个 epoch 的目标步数（通常等于 `max_episode_steps`）。 |
| `base_config_name` | `r1pro_behavior` | 在 `omni_config` 覆盖前加载的 OmniGibson 基础配置。 |

### `omni_config.task.*`（任务配置）

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `type` | `RLinfBehaviorTask` | **保持不变。** RLinf 修改版任务类。 |
| `activity_name` | `turning_on_radio` | 任务名称。更改此项以切换任务。 |
| `activity_definition_id` | `0` | 任务定义变体（通常为 0）。 |
| `activity_instance_id` | `0` | `mode=disabled` 下使用缓存目录时的实例 id。 |
| `instance_resample_mode` | `disabled` | `disabled` / `offline` / `online`。 |
| `activity_instance_dir` | `null` | 缓存实例文件路径（`offline` 模式必填）。 |
| `instance_file_format` | `tro_state` | `template` 或 `tro_state`。当 `activity_instance_dir` 设置时必填。 |
| `online_object_sampling` | `False` | 启用 BDDL 采样。`online` 模式必填。 |
| `use_presampled_robot_pose` | `True` | 使用缓存的机器人位姿。`online` 模式或 tro_state 缺少位姿时设为 `False`。 |
| `termination_config.max_steps` | `500` | OmniGibson 超时。**被 `max_episode_steps` 自动覆盖。** |
| `reward_config.r_potential` | `1.0` | 基于势函数的奖励系数。 |

### `omni_config.scene.*`（场景配置）

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `type` | `RLinfInteractiveTraversableScene` | **保持不变。** RLinf 修改版场景类。 |
| `scene_model` | `house_double_floor_lower` | 场景名称。必须与任务匹配。 |
| `load_room_types` | `["living_room", "kitchen"]` | 需加载的房间类型（优化加载时间）。 |
| `load_room_instances` | `null` | 特定房间实例。`null` 表示全部加载。 |

### `omni_config.env.*`（环境步进配置）

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `env_wrapper` | `rgb` | Wrapper 类型：`rgb`、`rgb_lowres`、`rich_obs`、`default`。 |
| `action_frequency` | `60` | 动作步进频率（Hz）。 |
| `physics_frequency` | `120` | 物理仿真频率（Hz）。必须 >= 动作频率。 |
| `rendering_frequency` | `60` | 渲染频率（Hz）。 |
| `automatic_reset` | `False` | **保持 False。** RLinf 自行管理重置。 |
| `flatten_obs_space` | `False` | **保持 False。** RLinf 需要结构化字典。 |
| `flatten_action_space` | `False` | **保持 False。** RLinf 需要结构化字典。 |

### `omni_config.camera.*`（相机分辨率）

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `head_resolution` | `[720, 720]` | ZED 相机分辨率。 |
| `wrist_resolution` | `[480, 480]` | RealSense 左/右腕部相机分辨率。 |

### `omni_config.macro.*`（性能开关）

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `use_gpu_dynamics` | `False` | GPU 物理（仅在涉及粒子/流体时启用）。 |
| `headless` | `True` | 无显示窗口。保持 `True`。 |
| `enable_flatcache` | `True` | USD flatcache 加速。保持 `True`。 |
| `enable_object_states` | `True` | **保持 True。** BehaviorTask 依赖此项。 |
| `enable_transition_rules` | `True` | 状态变化规则（切割、烹饪等）。 |
| `render_viewer_camera` | `False` | 向查看器相机渲染（开销大）。保持 `False`。 |
| `use_numpy_controller_backend` | `True` | Numpy 控制器后端（更快）。保持 `True`。 |

## 示例：多任务 offline 评估

要使用多样化场景评估多个任务：

```bash
# 任务 1：turning_on_radio（offline）
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_agentic \
  env.eval.omni_config.task.activity_name=turning_on_radio \
  env.eval.omni_config.task.instance_resample_mode=offline \
  env.eval.omni_config.task.activity_instance_dir=/path/to/instances/

# 任务 8：rearranging_kitchen_furniture（offline）
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_agentic \
  env.eval.omni_config.task.activity_name=rearranging_kitchen_furniture \
  env.eval.omni_config.task.instance_resample_mode=offline \
  env.eval.omni_config.task.activity_instance_dir=/path/to/instances/ \
  env.eval.omni_config.scene.load_room_types='["kitchen"]'
```

## 相关文档

- [vla_eval.md](vla_eval.md) -- 独立 VLA 评估指南
- [vlm_vla_eval.md](vlm_vla_eval.md) -- VLM+VLA 评估指南
- [data.md](data.md) -- 数据参考手册
