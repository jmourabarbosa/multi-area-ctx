from neurogym import spaces
import neurogym as ngym
import numpy as np

class GoNogoContextDecisionMaking2(ngym.TrialEnv):
    """Context-dependent decision-making task.

    The agent simultaneously receives stimulus inputs from two modalities (
    for example, a colored random dot motion pattern with color and motion
    modalities). The agent needs to make a perceptual decision based on
    only one of the two modalities, while ignoring the other. The relevant
    modality is explicitly indicated by a rule signal.
    """
    metadata = {
        'paper_link': 'https://www.nature.com/articles/nature12742',
        'paper_name': '''Context-dependent computation by recurrent
         dynamics in prefrontal cortex''',
        'tags': ['perceptual', 'context dependent', 'two-alternative',
                 'supervised']
    }

    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.0):
        super().__init__(dt=dt)

        # trial conditions
        self.contexts = [0, 1]  # index for context inputs
        self.choices = [1, 2]  # left, right choice
        self.cohs = [-1,1]
        self.sigma = sigma / np.sqrt(self.dt)  # Input noise

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            #'fixation': 300,
            'fixation': 300,
            # 'target': 350,
            'stimulus': 700,
            'delay': 0,
            'decision': 100}
        
        if timing:
            self.timing.update(timing)

        self.abort = False

        # set action and observation space
        names = ['stim1_mod1', 
                 'stim1_mod2', 'context1', 'context2']
        name = {name: i for i, name in enumerate(names)}
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(len(names),),
                                            dtype=np.float32, name=name)

        #name = {'fixation': 0, 'choice1': 1, 'choice2': 2}
        name = {'gono': 0, 'choice1': 1, 'choice2': 2}
        name = {'goleft': 0, 'goright':1}
        self.action_space = spaces.Discrete(2, name=name)

    def _new_trial(self, **kwargs):
        # -------------------------------------------------------------------------
        # Trial
        # -------------------------------------------------------------------------
        trial = {
            'ground_truth': 0,
            'context': self.rng.choice(self.contexts),
            'coh_0': self.rng.choice(self.cohs),
            'coh_1': self.rng.choice(self.cohs),
            'catch': self.rng.choice([0,0,0,1]),
        }
        trial.update(kwargs)

        if trial['context'] == 0:
          if trial['coh_0'] > 0: trial['ground_truth'] = 1
        if trial['context'] == 1:
           if trial['coh_1'] < 0: trial['ground_truth'] = -1
           
        # if trial["catch"] == 1:
        #     trial['coh_0'] = trial['coh_1'] = trial['ground_truth'] = 0
        
        # self.timing["fixation"] = int(self.rng.choice([200,300,400]))
        # self.timing["stimulus"] = int(1000 - self.timing["fixation"])
        
        # -----------------------------------------------------------------------
        # Periods
        # -----------------------------------------------------------------------
        periods = ['fixation', 'stimulus', 'delay', 'decision']
        self.add_period(periods)

        # self.add_ob(1, where='fixation')
        self.add_ob(trial['coh_0'], period='stimulus', where='stim1_mod1')
        self.add_ob(trial['coh_1'], period='stimulus', where='stim1_mod2')
        #self.add_randn(0, self.sigma, 'stimulus')
        self.set_ob(0, 'decision')

        if trial['context'] == 0:
            self.add_ob(1, where='context1')
        else:
            self.add_ob(1, where='context2')

        self.set_groundtruth(trial['ground_truth'], 'decision')
        
        self.set_groundtruth(trial['ground_truth'], 'stimulus')
        self.set_groundtruth(0, 'fixation')

        return trial

    def _step(self, action):
        ob = self.ob_now
        gt = self.gt_now

        new_trial = False
        reward = 0
        if self.in_period('fixation'):
            if action != 0:
                new_trial = self.abort
                reward = self.rewards['abort']
        elif self.in_period('decision'):
            if action != 0:  # broke fixation
                new_trial = True
                if action == gt:
                    reward = self.rewards['correct']
                    self.performance = 1

        return ob, reward, False, {'new_trial': new_trial, 'gt': gt}
        
class GoNogoContextDecisionMaking(ngym.TrialEnv):
    """Context-dependent decision-making task.

    The agent simultaneously receives stimulus inputs from two modalities (
    for example, a colored random dot motion pattern with color and motion
    modalities). The agent needs to make a perceptual decision based on
    only one of the two modalities, while ignoring the other. The relevant
    modality is explicitly indicated by a rule signal.
    """
    metadata = {
        'paper_link': 'https://www.nature.com/articles/nature12742',
        'paper_name': '''Context-dependent computation by recurrent
         dynamics in prefrontal cortex''',
        'tags': ['perceptual', 'context dependent', 'two-alternative',
                 'supervised']
    }

    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.0):
        super().__init__(dt=dt)

        # trial conditions
        self.contexts = [0, 1]  # index for context inputs
        self.choices = [1, 2]  # left, right choice
        self.cohs = [-1,1]
        self.sigma = sigma / np.sqrt(self.dt)  # Input noise

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': 4*dt,# 3
            'stimulus': 10*dt, # 
            'delay': 0,
            'decision': 0}
        
        if timing:
            self.timing.update(timing)

        self.abort = False

        # set action and observation space
        names = ['stim1_mod1', 
                 'stim1_mod2', 'context1', 'context2']
        name = {name: i for i, name in enumerate(names)}
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(len(names),),
                                            dtype=np.float32, name=name)

        #name = {'fixation': 0, 'choice1': 1, 'choice2': 2}
        name = {'gono': 0, 'choice1': 1, 'choice2': 2}
        name = {'gono': 0}
        self.action_space = spaces.Discrete(1, name=name)

    def _new_trial(self, **kwargs):
        # -------------------------------------------------------------------------
        # Trial
        # -------------------------------------------------------------------------
        trial = {
            'ground_truth': 0,
            'context': self.rng.choice(self.contexts),
            'coh_0': self.rng.choice(self.cohs),
            'coh_1': self.rng.choice(self.cohs),
            'catch': self.rng.choice([0,0,0,1]),
        }
        trial.update(kwargs)

        if trial['context'] == 0:
          if trial['coh_0'] < 0: trial['ground_truth'] = 1
        if trial['context'] == 1:
           if trial['coh_1'] < 0: trial['ground_truth'] = -1
           
        # if trial["catch"] == 1:
        #     trial['coh_0'] = trial['coh_1'] = trial['ground_truth'] = 0
        
        # self.timing["fixation"] = int(self.rng.choice([200,300,400]))
        # self.timing["stimulus"] = int(1000 - self.timing["fixation"])
        
        # -----------------------------------------------------------------------
        # Periods
        # -----------------------------------------------------------------------
        periods = ['fixation', 'stimulus', 'delay', 'decision']
        self.add_period(periods)

        # self.add_ob(1, where='fixation')
        self.add_ob(trial['coh_0'], period='stimulus', where='stim1_mod1')
        self.add_ob(trial['coh_1'], period='stimulus', where='stim1_mod2')
        self.add_randn(0, self.sigma, 'stimulus')
        #self.add_randn(0, self.sigma, 'fixation')

        self.set_ob(0, 'decision')

        if trial['context'] == 0:
            self.add_ob(1, where='context1')
        else:
            self.add_ob(1, where='context2')

        #self.set_groundtruth(trial['ground_truth'], 'decision')
        
        self.set_groundtruth(trial['ground_truth'], 'stimulus')
        self.set_groundtruth(trial['ground_truth'], 'delay')
        self.set_groundtruth(0, 'fixation')


        return trial

    def _step(self, action):
        ob = self.ob_now
        gt = self.gt_now

        new_trial = False
        reward = 0
        if self.in_period('fixation'):
            if action != 0:
                new_trial = self.abort
                reward = self.rewards['abort']
        elif self.in_period('decision'):
            if action != 0:  # broke fixation
                new_trial = True
                if action == gt:
                    reward = self.rewards['correct']
                    self.performance = 1

        return ob, reward, False, {'new_trial': new_trial, 'gt': gt}
     
def infer_test_timing(env):
  """Infer timing of environment for testing."""
  timing = {}
  for period in env.timing.keys():
      period_times = [env.sample_time(period) for _ in range(100)]
      timing[period] = np.median(period_times)
  return timing
  
# dt = 100
# n_trials_seq = 10
# envid = 'mycontext_no_input_training'
# env_kwargs = {'dt': dt}
# env = task.GoNogoContextDecisionMaking(**env_kwargs) 
# trial_dur = sum(list(env.timing.values())) // dt

# modelpath = get_modelpath(envid)
# config = {
#     'dt': dt,
#     'hidden_size': 3000,
#     'lr': 1e-3,
#     'batch_size': 16,
#     'seq_len': trial_dur * n_trials_seq,
#     'trial_dur': trial_dur,
#     'n_trials_seq': n_trials_seq,
#     'envid': envid,
#     'noise': 0,
# }

# config['env_kwargs'] = env_kwargs

# # Save config
# with open(modelpath / 'config.json', 'w') as f:
#     json.dump(config, f)
