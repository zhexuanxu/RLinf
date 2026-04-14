# BEHAVIOR-1K Configuration Reference

This document provides a comprehensive reference for configuring BEHAVIOR-1K tasks and scene initialization in RLinf.

## Supported Tasks (0–49)

BEHAVIOR-1K includes 50 household manipulation tasks. To switch tasks, set `omni_config.task.activity_name` in your config YAML.

| Index | `activity_name` | Description |
|-------|-----------------|-------------|
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

## Scene Initialization Modes

The `omni_config.task.instance_resample_mode` setting controls how scenes are initialized at the start of each episode.

### Mode 1: `disabled` (default) — Fixed Instance

```yaml
omni_config:
  task:
    instance_resample_mode: disabled
    activity_instance_id: 0
    activity_instance_dir: null
    online_object_sampling: False
    use_presampled_robot_pose: True
```

**Behavior**: Every episode resets to the exact same scene state.
- If `activity_instance_dir` is `null`, uses OmniGibson's built-in default instance.
- If `activity_instance_dir` is set, loads instance id=`activity_instance_id` from that directory.

#### Using a Specific `tro_state` Instance with `disabled` Mode

When `activity_instance_dir` points to a directory of `tro_state` files and you want to
evaluate on a specific instance (e.g., instance 242), you must also set `scene.scene_instance`
to an existing `_template.json` name. This is because OmniGibson requires a full template file
to build the scene during initialization, while the `tro_state` file only replaces
task-relevant object states after the scene is already loaded.

```yaml
omni_config:
  task:
    activity_name: turning_on_radio
    activity_instance_id: 242                     # Target instance for reset-time tro_state loading
    instance_resample_mode: disabled
    activity_instance_dir: /path/to/2025-challenge-task-instances/scenes/house_double_floor_lower/json/house_double_floor_lower_task_turning_on_radio_instances/
    instance_file_format: tro_state
    online_object_sampling: False
    use_presampled_robot_pose: True
  scene:
    type: RLinfInteractiveTraversableScene
    scene_model: house_double_floor_lower
    scene_instance: house_double_floor_lower_task_turning_on_radio_0_0_template  # Full template for init
```

**What happens**:
1. OmniGibson initializes the scene using the `scene_instance` template (`*_0_0_template.json`).
2. RLinf's `ActivityInstanceLoader` scans the `activity_instance_dir` for `tro_state` files.
3. On each reset, the loader applies the `tro_state` file for `activity_instance_id=242`,
   replacing task-relevant object positions and robot pose in-place.

**Why `scene_instance` is needed here but not with the default `activity_instance_id: 0`**:

When `scene_instance` is not set in the config, OmniGibson's `BehaviorTask.verify_scene_and_task_config()`
auto-generates it from `activity_instance_id`:

```
scene_instance = "{scene_model}_task_{activity}_{definition_id}_{instance_id}_template"
```

With the default `activity_instance_id: 0`, this produces
`house_double_floor_lower_task_turning_on_radio_0_0_template`, whose corresponding
`_template.json` exists in the `task_dir/json/` directory. So it works without explicitly
setting `scene_instance`.

When you change `activity_instance_id` to 242, the auto-generated name becomes
`*_0_242_template`, but the official dataset only provides `tro_state` files for
non-zero instance IDs (no `_template.json`). Setting `scene_instance` explicitly
overrides the auto-generation so OmniGibson loads the existing id=0 template instead.

### Mode 2: `offline` — Random Sampled from Cache

```yaml
omni_config:
  task:
    instance_resample_mode: offline
    activity_instance_dir: /path/to/cached_instances/  # REQUIRED
    instance_file_format: tro_state  # or "template"
    online_object_sampling: False
    use_presampled_robot_pose: True
```

**Behavior**: At startup, RLinf scans `activity_instance_dir`. Before each reset, one instance is randomly sampled.

Two formats:
- **`tro_state`**: Lightweight, restores task-relevant objects + robot poses in-place
- **`template`**: Full scene reload, complete reset but slower

#### Using `offline` with `tro_state` Files

When using `tro_state` format in `offline` mode, you must also set `scene.scene_instance`
for the same reason as in `disabled` mode: OmniGibson needs a full template to initialize
the scene, and `tro_state` files can only replace object states after initialization.

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

**What happens**:
1. OmniGibson initializes the scene using the `scene_instance` template.
2. RLinf's `ActivityInstanceLoader` scans `activity_instance_dir` for all `tro_state` files.
3. On each reset, one instance is chosen via `random.choice()` and its tro_state is applied
   in-place, replacing task-relevant object positions and robot pose.

Note: `activity_instance_id` is ignored in `offline` mode (the instance is always random).
If `instance_file_format` is `template` instead of `tro_state`, `scene_instance` is not
required because each template file contains the full scene definition.

### Mode 3: `online` — Live BDDL Sampling

```yaml
omni_config:
  task:
    instance_resample_mode: online
    online_object_sampling: True
    use_presampled_robot_pose: False
    activity_instance_dir: null  # must NOT be set
```

**Behavior**: Before each reset, OmniGibson's BDDL constraint solver generates a new scene. Most diverse but slowest.

## Downloading and Generating Initialization Samples

### Official Challenge Instances

```bash
export OMNIGIBSON_DATA_PATH=/path/to/BEHAVIOR-1K-datasets
python -c "from omnigibson.utils.asset_utils import download_2025_challenge_task_instances; download_2025_challenge_task_instances()"
```

Downloads to `$OMNIGIBSON_DATA_PATH/og_dataset/2025-challenge-task-instances/<scene_model>/json/<scene_model>_task_<activity_name>_instances/`

### RLinf Instance Generator

```bash
cd /path/to/RLinf

# Generate tro_state instances (recommended)
python rlinf/envs/behavior/instance_generator.py \
  --config examples/embodiment/config/env/behavior_r1pro.yaml \
  --output-format tro_state \
  --start-idx 1 \
  --end-idx 50

# Generate full-template instances
python rlinf/envs/behavior/instance_generator.py \
  --config examples/embodiment/config/env/behavior_r1pro.yaml \
  --output-format template \
  --start-idx 1 \
  --end-idx 50
```

## Configuration Reference

### Top-level RLinf Fields

| Field | Default | Meaning |
|-------|---------|---------|
| `env_type` | `behavior` | Environment type identifier. |
| `auto_reset` | `False` | Auto-reset done envs within epoch. Set `True` for eval-only with `ignore_terminations=True`. |
| `ignore_terminations` | `False` | Treat all episode ends as truncations (no task completion signals). |
| `use_rel_reward` | `True` | Use relative/potential-based reward. |
| `max_episode_steps` | `2000` | Episode hard limit (RLinf enforces at exact step; OmniGibson has off-by-one). |
| `max_steps_per_rollout_epoch` | `2000` | Target steps per epoch (usually equals `max_episode_steps`). |
| `base_config_name` | `r1pro_behavior` | OmniGibson base config to load before `omni_config` overrides. |

### `omni_config.task.*` (Task Configuration)

| Field | Default | Meaning |
|-------|---------|---------|
| `type` | `RLinfBehaviorTask` | **Keep as-is.** RLinf-patched task class. |
| `activity_name` | `turning_on_radio` | Task name. Change to switch tasks. |
| `activity_definition_id` | `0` | Task definition variant (usually 0). |
| `activity_instance_id` | `0` | Instance id for `mode=disabled` with cached dir. |
| `instance_resample_mode` | `disabled` | `disabled` / `offline` / `online`. |
| `activity_instance_dir` | `null` | Path to cached instance files (required for `offline`). |
| `instance_file_format` | `tro_state` | `template` or `tro_state`. Required when `activity_instance_dir` is set. |
| `online_object_sampling` | `False` | Enable BDDL sampling. Required for `online`. |
| `use_presampled_robot_pose` | `True` | Use cached robot poses. Set `False` for `online` or when tro_state lacks poses. |
| `termination_config.max_steps` | `500` | OmniGibson timeout. **Auto-overridden by `max_episode_steps`.** |
| `reward_config.r_potential` | `1.0` | Potential-based reward coefficient. |

### `omni_config.scene.*` (Scene Configuration)

| Field | Default | Meaning |
|-------|---------|---------|
| `type` | `RLinfInteractiveTraversableScene` | **Keep as-is.** RLinf-patched scene class. |
| `scene_model` | `house_double_floor_lower` | Scene name. Must match task. |
| `load_room_types` | `["living_room", "kitchen"]` | Which rooms to load (optimize load time). |
| `load_room_instances` | `null` | Specific room instances. `null` = all. |

### `omni_config.env.*` (Environment Step Configuration)

| Field | Default | Meaning |
|-------|---------|---------|
| `env_wrapper` | `rgb` | Wrapper type: `rgb`, `rgb_lowres`, `rich_obs`, `default`. |
| `action_frequency` | `60` | Action stepping frequency (Hz). |
| `physics_frequency` | `120` | Physics simulation frequency (Hz). Must be ≥ action frequency. |
| `rendering_frequency` | `60` | Rendering frequency (Hz). |
| `automatic_reset` | `False` | **Keep False.** RLinf manages resets. |
| `flatten_obs_space` | `False` | **Keep False.** RLinf expects structured dicts. |
| `flatten_action_space` | `False` | **Keep False.** RLinf expects structured dicts. |

### `omni_config.camera.*` (Camera Resolution)

| Field | Default | Meaning |
|-------|---------|---------|
| `head_resolution` | `[720, 720]` | ZED camera resolution. |
| `wrist_resolution` | `[480, 480]` | RealSense L/R camera resolution. |

### `omni_config.macro.*` (Performance Toggles)

| Field | Default | Meaning |
|-------|---------|---------|
| `use_gpu_dynamics` | `False` | GPU physics (enable only for particles/fluids). |
| `headless` | `True` | No viewer window. Keep `True`. |
| `enable_flatcache` | `True` | USD flatcache for speed. Keep `True`. |
| `enable_object_states` | `True` | **Keep True.** BehaviorTask requires this. |
| `enable_transition_rules` | `True` | State-change rules (slicing, cooking, etc.). |
| `render_viewer_camera` | `False` | Render to viewer camera (expensive). Keep `False`. |
| `use_numpy_controller_backend` | `True` | Numpy controller (faster). Keep `True`. |

## Example: Multi-Task Offline Evaluation

To evaluate multiple tasks with diverse scenes:

```bash
# Task 1: turning_on_radio (offline)
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_agentic \
  env.eval.omni_config.task.activity_name=turning_on_radio \
  env.eval.omni_config.task.instance_resample_mode=offline \
  env.eval.omni_config.task.activity_instance_dir=/path/to/instances/

# Task 8: rearranging_kitchen_furniture (offline)
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_agentic \
  env.eval.omni_config.task.activity_name=rearranging_kitchen_furniture \
  env.eval.omni_config.task.instance_resample_mode=offline \
  env.eval.omni_config.task.activity_instance_dir=/path/to/instances/ \
  env.eval.omni_config.scene.load_room_types='["kitchen"]'
```

## See Also

- [vla_eval.md](vla_eval.md) — Standalone VLA evaluation guide
- [vlm_vla_eval.md](vlm_vla_eval.md) — VLM+VLA evaluation guide
