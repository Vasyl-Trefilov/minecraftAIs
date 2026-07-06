import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=2048)]
        )
        print("Set GPU memory limit to 2048MB")
    except RuntimeError as e:
        print("GPU Memory Config Error:", e)

@tf.keras.utils.register_keras_serializable()
class AddCoords(layers.Layer):
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        h = 180
        w = 320

        y_coords = tf.linspace(-1.0, 1.0, h)
        x_coords = tf.linspace(-1.0, 1.0, w)
        x_grid, y_grid = tf.meshgrid(x_coords, y_coords)

        x_grid = tf.expand_dims(tf.expand_dims(x_grid, axis=0), axis=-1)
        y_grid = tf.expand_dims(tf.expand_dims(y_grid, axis=0), axis=-1)

        x_grid = tf.tile(x_grid, [batch_size, 1, 1, 1])
        y_grid = tf.tile(y_grid, [batch_size, 1, 1, 1])

        return tf.concat([inputs, x_grid, y_grid], axis=-1)

try:
    import gym
    import minerl
    HAS_MINERL = True
    print("Successfully imported MineRL!")
except ImportError:
    HAS_MINERL = False
    import gym
    print("Warning: minerl not found. A Mock environment will be used.")

class MockMinecraftEnv(gym.Env):
    def __init__(self):
        super(MockMinecraftEnv, self).__init__()
        self.observation_space = gym.spaces.Dict({
            'pov': gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        })
        self.action_space = gym.spaces.Dict({
            'forward': gym.spaces.Discrete(2),
            'left': gym.spaces.Discrete(2),
            'back': gym.spaces.Discrete(2),
            'right': gym.spaces.Discrete(2),
            'attack': gym.spaces.Discrete(2),
            'camera': gym.spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32)
        })
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return {'pov': np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)}

    def step(self, action):
        self.step_count += 1
        
        reward = 0.0
        if action['forward'] == 1:
            reward += 0.05
        if action['attack'] == 1:
            reward += 0.2
            
        done = self.step_count >= 120
        obs = {'pov': np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)}
        return obs, reward, done, {}

MODEL_PATH = "chop_bot_v1static.keras"
FRAME_STACK = 4
IMG_H, IMG_W = 180, 320

if os.path.exists(MODEL_PATH):
    print(f"Loading pre-trained BC model: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
else:
    raise FileNotFoundError(f"Error: {MODEL_PATH} not found. Please train BC weights first.")

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)

def preprocess_pov(pov_img):
    resized = cv2.resize(pov_img, (IMG_W, IMG_H))
    return resized.astype(np.float32) / 255.0

def sample_action(keyboard_pred, mouse_pred, exploration_noise=0.15):
    sampled_keys = []
    for prob in keyboard_pred:
        prob = np.clip(prob, 1e-7, 1.0 - 1e-7)
        key_act = np.random.binomial(1, prob)
        sampled_keys.append(key_act)
        
    sampled_mouse = mouse_pred + np.random.normal(0, exploration_noise, size=2)
    sampled_mouse = np.clip(sampled_mouse, -1.0, 1.0)
    
    return np.array(sampled_keys, dtype=np.float32), np.array(sampled_mouse, dtype=np.float32)

@tf.function
def train_step(states, actions_keys, actions_mouse, discounted_returns):
    states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
    actions_keys_tensor = tf.convert_to_tensor(actions_keys, dtype=tf.float32)
    actions_mouse_tensor = tf.convert_to_tensor(actions_mouse, dtype=tf.float32)
    discounted_returns_tensor = tf.convert_to_tensor(discounted_returns, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        keyboard_pred, mouse_pred = model(states_tensor, training=True)
        
        keyboard_pred = tf.clip_by_value(keyboard_pred, 1e-7, 1.0 - 1e-7)
        log_prob_keys = actions_keys_tensor * tf.math.log(keyboard_pred) + (1.0 - actions_keys_tensor) * tf.math.log(1.0 - keyboard_pred)
        log_prob_keys = tf.reduce_sum(log_prob_keys, axis=-1)
        
        sigma = 0.15
        const_term = tf.math.log(tf.constant(sigma * np.sqrt(2 * np.pi), dtype=tf.float32))
        log_prob_mouse = -0.5 * tf.square((actions_mouse_tensor - mouse_pred) / sigma) - const_term
        log_prob_mouse = tf.reduce_sum(log_prob_mouse, axis=-1)
        
        total_log_prob = log_prob_keys + log_prob_mouse
        
        loss = -tf.reduce_mean(total_log_prob * discounted_returns_tensor)
        
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

def compute_discounted_returns(rewards, gamma=0.99):
    discounted = np.zeros_like(rewards, dtype=np.float32)
    cumulative = 0.0
    for i in reversed(range(len(rewards))):
        cumulative = rewards[i] + cumulative * gamma
        discounted[i] = cumulative
        
    mean = np.mean(discounted)
    std = np.std(discounted) + 1e-8
    return (discounted - mean) / std

def run_rl_training(episodes=50, max_steps=300):
    if HAS_MINERL:
        print("Launching MineRL environment...")
        env = gym.make('MineRLObtainDiamond-v0')
    else:
        env = MockMinecraftEnv()

    print(f"\nStarting RL Fine-Tuning for {episodes} episodes...")
    for ep in range(episodes):
        obs = env.reset()
        pov = preprocess_pov(obs['pov'])
        
        frame_buffer = [pov] * FRAME_STACK
        
        states = []
        actions_k = []
        actions_m = []
        rewards = []
        
        episode_reward = 0
        
        for step in range(max_steps):
            stacked_state = np.concatenate(frame_buffer, axis=-1)
            states.append(stacked_state)
            
            input_tensor = np.expand_dims(stacked_state, axis=0)
            k_pred, m_pred = model(input_tensor, training=False)
            k_pred = k_pred.numpy()[0]
            m_pred = m_pred.numpy()[0]
            
            keys_act, mouse_act = sample_action(k_pred, m_pred)
            
            actions_k.append(keys_act)
            actions_m.append(mouse_act)
            
            env_action = {
                'forward': int(keys_act[0]),
                'left': int(keys_act[1]),
                'back': int(keys_act[2]),
                'right': int(keys_act[3]),
                'attack': int(keys_act[4]),
                'camera': mouse_act * 8.0 
            }
            
            next_obs, reward, done, _ = env.step(env_action)
            rewards.append(reward)
            episode_reward += reward
            
            next_pov = preprocess_pov(next_obs['pov'])
            frame_buffer.append(next_pov)
            frame_buffer.pop(0)
            
            if done:
                break
                
        discounted_returns = compute_discounted_returns(rewards)
        loss = train_step(states, actions_k, actions_m, discounted_returns)
        
        print(f"Episode {ep+1:03d}/{episodes:03d} | Total Reward: {episode_reward:6.2f} | Policy Loss: {loss:8.4f}")
        
        if (ep + 1) % 5 == 0:
            rl_model_path = "chop_bot_rl.keras"
            model.save(rl_model_path)
            print(f"--> Saved RL fine-tuned model to: {rl_model_path}")
            
    env.close()
    print("\nRL Training Complete!")

if __name__ == "__main__":
    run_rl_training(episodes=50, max_steps=300)
