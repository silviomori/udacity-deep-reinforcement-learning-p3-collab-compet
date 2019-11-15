class Config:
    def __new__(self):
        """Define this class as a singleton"""
        if not hasattr(self, 'instance'):
            self.instance = super().__new__(self)

            self.instance.device = None
            self.instance.seed = None

            self.instance.state_size = None
            self.instance.action_size = None
            self.instance.num_agents = None

            self.instance.actor_layers = None
            self.instance.critic_layers = None
            self.instance.actor_lr = None
            self.instance.critic_lr = None

        return self.instance
