import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython.display import display, clear_output
import random
#tsp brute force
import itertools
import json

class CustomGridEnv(gym.Env):
    def __init__(self, grid_size=[10, 10],start_positions=[(0,0)],goal_sets=[[(9,9)]]):
        super(CustomGridEnv, self).__init__()
        self.isLearning=True
        self.grid_size = grid_size
        self.start_positions = start_positions
        self.goal_sets = goal_sets
        self.observation_space = spaces.Discrete(np.prod(grid_size))
        self.action_space = spaces.Discrete(4)  # 4 possible actions: 0 - Up, 1 - Down, 2 - Left, 3 - Right
        self.goal_orders=[self.travelling_salesman(goal_positions,start_position) for goal_positions,start_position in zip(goal_sets,self.start_positions)]
        self.path=[]
        self.prev_action=5
        self.paths=[]
        self.current_goal_set=0
        self.successful_attempts=0
        self.reset()
        self.actions_taken=[]#start
        # self.d=defaultdict(lambda:0)
        self.default_goal_orders=[self.indexes(goal_positions) for goal_positions in goal_sets]
    def learn(self,num_episodes=100,verbose=False,doRender=False,iterations=1,learning_rate=0.1,discount_factor= 0.8,exploration_rate= 1.0,max_exploration_rate=1.0,min_exploration_rate=0.01,exploration_decay_rate= 0.01,max_steps_per_episode = 100):
        self.max_steps_per_episode = max_steps_per_episode
        self.exploration_rate=exploration_rate
        self.Q1=[]
        s=len(self.goal_sets) #no of sets
        #--------------#
        for goal_set in range(s):
                  self.done=False
                  self.current_goal_set=goal_set
                  n=len(self.goal_sets[self.current_goal_set]) #no of goals
                  Q = np.array([np.zeros((self.grid_size[0], self.grid_size[1], 4 ))]*n)
                  #-----------------#
                  self.E_rewards=[]
                  for goal in range(n):
                          self.E_rewards=[]
                          #----Q-learning Algorithm----#
                          for iter in range(iterations):
                                  self.successful_attempts = 0
                                  self.Goal_rewards=[]
                                  exploration_rate=self.exploration_rate
                                  # self.d=defaultdict(lambda:0)
                                  for episode in range(num_episodes):
                                            #---------starting from one end
                                            #state = self.start_position
                                            #-----------starting from any random position
                                            #state=(spaces.Discrete(grid_size[0]).sample(),spaces.Discrete(grid_size[0]).sample())
                                            pos = [0,self.grid_size[0] - 1 , (self.grid_size[0]) // 2]# for 10X10 [0,9,5]
                                            state = (random.choice(pos),random.choice(pos))
                                            # self.d[state]+=1
                                            done = False
                                            total_reward = 0
                                            self.reset()
                                            self.current_position=state
                                            for steps in range(max_steps_per_episode):
                                                      #------------Exploration-exploitation tradeoff
                                                      exploration_threshold = np.random.uniform(0,1)
                                                      v_actions=self.valid_actions()# valid actions for current posito
                                                      if exploration_threshold > exploration_rate:
                                                        action = np.argmax(Q[goal,state[0], state[1]])
                                                        #-------avoiding invalid actions while learning
                                                        if(action not in v_actions):
                                                            Q[goal,state[0], state[1],action]= -1e9
                                                            action = np.argmax(Q[goal,state[0], state[1]])
                                                      else:
                                                        try:
                                                            v_actions.remove(self.prev)
                                                        except:
                                                            k=0
                                                        action=random.sample(v_actions,1)[0]
                                                        # action= np.random.randint(0,4)
                                                        # while(action not in v_actions):
                                                        #     action= np.random.randint(0,4)
                                                      self.prev_action=action
                                                      #-----------Take the chosen action and observe the new stsate and reward
                                                      new_state,reward,done,_=self.step(action,goal)

                                                      #-----------Update Q-value using bellman eqn
                                                      if not( new_state==state):
                                                          Q[goal,state[0], state[1], action] = Q[goal,state[0],state[1], action] + learning_rate*( reward + discount_factor * np.max(Q[goal,new_state[0], new_state[1]]) - Q[goal,state[0], state[1], action] )

                                                      total_reward += reward
                                                      state = new_state
                                                      if self.done:
                                                        break
                                                      if doRender :
                                                        self.render(E=episode,S=self.successful_attempts)
                                            if  ( not doRender) and verbose:
                                                print(f"goal {goal+1}, Episode {episode+1}/{num_episodes}, steps taken:{self.steps_taken}, Total Reward : {total_reward}")
                                            self.Goal_rewards.append(total_reward)
                                            #-----------Exploration rate decay
                                            exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
                                            #successful_attempts += total_reward
                                  # print(f"goal_set={goal_set+1},goal {goal+1}, iteration:{iter+1} succesful attempts:{self.successful_attempts}/{num_episodes}")
                                  self.E_rewards.append(self.Goal_rewards)
                  self.Q1.append(Q)
    def run(self,start=None,tsp=True,render=True):
      #-----------method to use q table to render output
        self.reset()
        self.paths=[]
        self.current_goal_set=0
        self.isLearning=False
        if start!=None:
            state=start
            self.start_positions[0]=state
            self.goal_order=self.travelling_salesman(self.goal_sets[self.current_goal_set],state)
        else:
            state=self.start_positions[0]
        self.path.append(state)
        render and self.render()
        if (tsp):
            current_orders=self.goal_orders
        else:
            current_orders=self.default_goal_orders
        for l in range(len(self.goal_sets)):
            self.current_goal_set=l
            self.reset()
            self.current_position=self.start_positions[l]
            state=self.current_position

            self.path=[state]
            for i in current_orders[self.current_goal_set]:
                self.done=False
                prev_state=None
                while(not self.done) and self.steps_taken<=self.max_steps_per_episode:
                    action=np.argmax(self.Q1[self.current_goal_set][i][state[0]][state[1]])#from q table
                    next_state=self.nxt_state(state,action)
                    if(tsp):
                      if ((not (0 <= next_state[0] < self.grid_size[0] and 0 <= next_state[1] < self.grid_size[1])) or not ( True if prev_state==None else (prev_state!=next_state)) or next_state in self.path  ):
                        #---------invalid moves
                        # --------moving outside the grid
                        # --------back and forth loop
                          self.Q1[self.current_goal_set][i][state[0]][state[1]][action]= 0
                          action=np.argmax(self.Q1[self.current_goal_set][i][state[0]][state[1]])

                    else:
                      if ((not (0 <= next_state[0] < self.grid_size[0] and 0 <= next_state[1] < self.grid_size[1])) or not ( True if prev_state==None else (prev_state!=next_state)) ):
                        #---------invalid moves
                        # --------moving outside the grid
                        # --------back and forth loop
                          self.Q1[self.current_goal_set][i][state[0]][state[1]][action]= 0
                          action=np.argmax(self.Q1[self.current_goal_set][i][state[0]][state[1]])

                    prev_state=state
                    state,reward,self.done,_=self.step(action,i)
                    if state not in self.path:
                        self.path.append(state)
                    render and self.render(g = i)

                if(self.steps_taken>self.max_steps_per_episode):
                    break
            self.paths.append(self.path)

            render and self.render(last=1,g="all")
        return self.json_paths()
    def _position_to_observation(self, position):
        # Convert the 2D position to a 1D observation
        return position[0] * self.grid_size[1] + position[1]
    def _observation_to_position(self,observation):
        return (observation//self.grid_size[1],observation % self.grid_size[1])
    def reset(self):
        self.current_position = self.start_positions[self.current_goal_set]
        # self.current_position=[0,0]
        self.steps_taken = 0
        # self.current_goal_set=0
        self.action_taken=[]
        self.done = False
        self.path=[]
        self.prev_action=5
        return self.current_position

    def step(self, action,goal):
        if self.done:
            raise ValueError("Episode has ended. Please call reset() to start a new episode.")
        next_state=self.nxt_state(self.current_position,action)
        #--------Check if the next state is within the boundaries
        #--------If the next state is outside the boundaries, stay in the current state
        if 0 <= next_state[0] < self.grid_size[0] and 0 <= next_state[1] < self.grid_size[1]:
            self.current_position=next_state
        self.action_taken.append(action)
        self.steps_taken += 1
        #---------Check if the agent has reached the goal
        if self.current_position == self.goal_sets[self.current_goal_set][goal]:
            self.done = True
            reward =1
            self.successful_attempts+=1
        else:
            reward = 0
        # reward=1/(1+self.distance(self.current_position,self.goal_sets[self.current_goal_set][goal]))
        #---------Define episode termination conditions
        if self.steps_taken >= self.max_steps_per_episode:
            self.done = True

        return self.current_position, reward,self.done, {}

    def render(self,E=0,S=0,last=0,g=0):
        # Visual rendering using matplotlib
        fig, ax = plt.subplots()
        ax.set_xticks(range(self.grid_size[1] + 1))
        ax.set_yticks(range(self.grid_size[0] + 1))

        # Add gridlines
        ax.grid(which='both')

        # Add patches for the agent and goal
        agent_patch = patches.Rectangle((self.current_position[1], self.current_position[0]),
                                        1, 1, linewidth=1, edgecolor='r', facecolor='r')
        #goal patches
        for goal_position in self.goal_sets[self.current_goal_set]:
          ax.add_patch(patches.Rectangle((goal_position[1], goal_position[0]),
                                       1, 1, linewidth=1, edgecolor='g', facecolor='g'))
        if(self.isLearning):
            ax.set_title(f"episode:{E+1},successful attempts by now:{S}")
        else:
            ax.set_title(f"Using Q-Table for set={self.current_goal_set+1},steps={self.steps_taken},goal={g}")
        if len(self.path):
            k=0.25/len(self.path)
            R=0
            for p in self.path:
                R+=k
                ax.add_patch(patches.Circle((p[1]+0.5,[p[0]+0.5]),R, color='blue'))
        ax.add_patch(agent_patch)
        # ax.add_patch(goal_patch)

        # Set axis limits
        ax.set_xlim(0, self.grid_size[1])
        ax.set_ylim(0, self.grid_size[0])

        display(fig)
        clear_output(wait=True)
        #self.wait()
        if not last:
          plt.close(fig)
    def distance(self,city1, city2):
        # Calculate Manhattan distance between two cities
        return abs(city1[0] - city2[0]) + abs(city1[1] - city2[1])

    def travelling_salesman(self,cities, start):
        #-----------------brute force-------------------
        cities=[start]+cities
        num_cities = len(cities)
        # print(f"Number of cities:{num_cities}")
        all_cities = set(range(num_cities))
        start_index = cities.index(start)
        all_cities.remove(start_index)
        min_distance = float('inf')
        min_path = None
        for perm in itertools.permutations(all_cities):
            path = [start_index] + list(perm) + [start_index]
            total_distance = 0
            for i in range(num_cities - 1):
                total_distance += self.distance(cities[path[i]], cities[path[i + 1]])

            if total_distance < min_distance:
                min_distance = total_distance
                min_path = path

        return [x-1 for x in min_path[1:-1]]

    def nxt_state(self,state,action):
        #----------------returns next state based on current state and action
        if action == 0:#down
            next_state = (state[0] - 1, state[1])
        elif action == 1:#up
            next_state = (state[0] + 1, state[1])
        elif action == 2:#left
            next_state = (state[0], state[1] - 1)
        else:#right
            next_state = (state[0], state[1] + 1)
        return next_state
    def opposite_action(self,a):
        if a==0:
          return 1
        if a==1:
          return 0
        if a==2:
          return 3
        if a==3:
          return 2
        else:
          return a
    def valid_actions(self):
        #-----------------uses self.current_position to return a tuple with valid actions
        up , down ,left ,right=0,1,2,3
        (x,y)=self.current_position
        res=[]
        if y==0:
            if x==0:
                res= [down,right]
            if x==self.grid_size[0]-1:
                res= [up,right]
            else:
                res= [down,up,right]
        elif y==self.grid_size[1]-1:
            if x==0:
                res= [down,left]
            if x==self.grid_size[0]-1:
                res= [up,left]
            else:
                res= [up,left,down]
        else:
            if x==0:
                res= [left,down,right]
            if x==self.grid_size[0]-1:
                res= [up,right,left]
            else:
                res= [up,down,left,right]
        # try:
        #    if (self.prev_action!=None):
        #         res.remove(self.opposite_action(self.prev_action))
        #    return res
        # except:
        #    return res
        return res
    def indexes(self,x):
        res=[]
        i=0
        for k in x:
          res.append(i)
          i+=1
        return res
    def get_direction(self,current_coord, next_coord):
        if current_coord[0] < next_coord[0]:
            return "D"#changed - tarun
        elif current_coord[0] > next_coord[0]:
            return "U"#changed - tarun
        elif current_coord[1] < next_coord[1]:
            return "R"#changed - tarun
        elif current_coord[1] > next_coord[1]:
            return "L"#changed - tarun
        else:
            return None
    def json_paths(self):
        json_data = []  # Initialize an empty list to store the JSON data

        for i, path in enumerate(self.paths): # Initialize an empty list for the current path
            path_data = dict() #changed the code in this for loop - tarun
            for j, coord in enumerate(path):
                if j == len(path) - 1:
                    path_data[f"{coord[0]},{coord[1]}"] = "END"
                else:
                    next_coord = path[j + 1]
                    direction = self.get_direction(coord, next_coord)
                    path_data[f"{coord[0]},{coord[1]}"] = direction

            json_data.append({"path": path_data})

        # Convert the list of dictionaries to a JSON-formatted string with indentation
        json_output = json.dumps(json_data, indent=2)

        # Return the JSON string
        return json_output
def run_rl(start_positions,goal_sets,grid_size):
  num_episodes = 20*grid_size[0]*grid_size[1]#predefined
  max_steps_per_episode = 50*grid_size[0]*grid_size[1] #larger the grid size larger the number of learning steps needed#predefined
  learning_rate=0.1#predefined
  discount_factor=0.96#predefined
  exploration_rate=1.0#predefined
  max_exploration_rate=1.0#predefined
  min_exploration_rate=0.01#predefined
  exploration_decay_rate= 0.8/num_episodes#predefined
  env=CustomGridEnv(grid_size=grid_size,start_positions=start_positions,goal_sets=goal_sets)
  print("Learning goals")
  env.learn(num_episodes=num_episodes ,verbose=False,iterations=5,doRender=False,learning_rate=learning_rate , discount_factor= discount_factor , exploration_rate= exploration_rate , max_exploration_rate=max_exploration_rate , min_exploration_rate=min_exploration_rate , exploration_decay_rate= exploration_decay_rate , max_steps_per_episode =max_steps_per_episode)
  print("Generating path")
  return env.run(tsp=True,render=False)