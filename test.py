import random
import envpool
import numpy as np
def test():
    #envpool.make("Breakout-v5",)
    pool = envpool.make(
            task_id="Breakout-v5",
            env_type='gym',
            num_envs=4,
            frame_skip=4,
            episodic_life=True,
            stack_num=4,
            noop_max=30,
            seed=random.randint(0, 10000),
            repeat_action_probability=.0,
            # batch_size=self.batch_size,
            # num_threads=self.size,
            # thread_affinity_offset=-1,
            # max_episode_steps=108000,
        )
    #print(pool.spec.lives)
    pool.send(np.array([0,0,0,0]))
    res=pool.recv()
    print(res[1:])


if __name__ == "__main__":
    test()

