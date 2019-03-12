[Report file in better presented PDF](report/Report.pdf)
---
author:
- 'Jung, Myoungki'
bibliography:
- 'bib.bib'
title: Deep Neural Network Reinforcement Learning of an Robot arm 
---

Introduction 
============

learning always has been in the scholars hot topic well before
Deepminds’ atari game mastering [@Mirowski2016], even before the TD
Gammon game shipped in a desktop. With advancement of deep neaural
networks, its application to reinforcment learning enables researchers
to overcome its previous theoretical limits and chalanges, high
dimensional problems, speed of convergence and many more, which are the
reasons why RL has been in research feild for a long time but not much
progress like other topics until recently. Alphago Zero[@Hassabis2017],
an agent learning the game of Go without any human game play data unlike
its predecessor, shows how eficient reinforcement learning can be
compared to humans’ learning by mastering the game within months and
play more intellegent than any human gamer in the world in that short
time.

Implementation
==============

Set up
------

Reward functions were coded in the cpp plugin file as shown in the
listing \[list:RewardFunction0\] and
\[list:IntermediaryRewardFunction\]. The contact were determined by the
`ArmPlugin::onCollisionMsg`, a callback function of contact subscription
shown in listing \[list:Load\].

    void ArmPlugin::Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/)
    {
          printf("ArmPlugin::Load('%s')\n", _parent->GetName().c_str());
          
          //pointer to the model
          this->model = _parent;
          this->j2_controller = new physics::JointController(model);
          
          // camera communication node
          cameraNode->Init();
          cameraSub = cameraNode->Subscribe("/gazebo/" WORLD_NAME "/camera/link/camera/image", &ArmPlugin::onCameraMsg, this);
          
          // collision detection node
          collisionNode->Init();
          collisionSub = collisionNode->Subscribe("/gazebo/" WORLD_NAME "/" PROP_NAME "/tube_link/my_contact", &ArmPlugin::onCollisionMsg, this);
          
          // event update
          this->updateConnection = event::Events::ConnectWorldUpdateBegin(boost::bind(&ArmPlugin::OnUpdate, this, _1));
    }

Reward function
---------------

The message from contacts topic was decoded and compared by `strcmp`
function and determined the reward or penalty. If contacted items are
tube and `gripper_link`, then the agent is rewared 12 points and other
cases the agents penalised for -8 points.

    // Define Reward Parameters
    #define REWARD_WIN   12.0f
    #define REWARD_LOSS -8.0f
    #define REWARD_MULTIPLIER 10.0f

    #define COLLISION_FILTER "ground_plane::link::collision"
    #define COLLISION_ITEM   "tube::tube_link::tube_collision"
    #define COLLISION_POINT  "arm::gripperbase::gripper_link"

    // onCollisionMsg
    void ArmPlugin::onCollisionMsg(ConstContactsPtr &contacts)
    {
          if( testAnimation ) return;
          
          for (int i = 0; i < contacts->contact_size(); ++i)
          {
                if( strcmp(contacts->contact(i).collision2().c_str(), COLLISION_FILTER) == 0 ) continue;
                
                std::cout << "Collision between[" << contacts->contact(i).collision1() << "] and [" << contacts->contact(i).collision2() << "]\n";
                
                if((strcmp(contacts->contact(i).collision1().c_str(), COLLISION_ITEM) == 0) && (strcmp(contacts->contact(i).collision2().c_str(), COLLISION_POINT) == 0))
                {
                rewardHistory = REWARD_WIN;
                      newReward  = true;
                      endEpisode = true;
                      return;
                }
                else {
                      rewardHistory = REWARD_LOSS;
                      newReward  = true;
                      endEpisode = true;
                }
          }
    }

The agent was rewared only once in the episode when its body part
contacts the object. The penalty condition was when the agent touches
the ground. On both contacts the episode was reset and restarts another
one immediately. The Reward function did not differenciate the task into
touching with body and touching with gripper. Because the gripper is the
end node frequently touch the object as robot’s physical build, Robot
tends to master on how to touch the object with gripper every time
traning it. In addition, it masters, more than 80 percent for 100
episoides, the task within 400 iterations. If the reward function sets
up with cases contacting body and the tube for less rewards, less than
12 points, it could learn to touch the tube with other parts than the
gripper link. However, this experiment was kept to simple to see the
application of DQN to RL for Robotic arm.

Intermediary Reward
-------------------

Listing \[list:IntermediaryRewardFunction\] shows how the Intermediary
Reward was calculated after timeout of an episode. This intermediary
reward is important to provide feedbacks to the agent how well it went
although it could not finish the task in a given time. The balance
between time penalty and reward in relation to the distance to the tube
is the key to provide a firm guideline for learning what is a policy the
user wants to teach to the agent.

    // if an EOE reward hasn't already been issued, compute an intermediary reward
    if( hadNewState && !newReward )
    {
          PropPlugin* prop = GetPropByName(PROP_NAME);
          
          if( !prop )
          {
                printf("ArmPlugin - failed to find Prop '%s'\n", PROP_NAME);
                return;
          }
          
          // get the bounding box for the prop object
          const math::Box& propBBox = prop->model->GetBoundingBox();
          physics::LinkPtr gripper  = model->GetLink(GRIP_NAME);
          
          if( !gripper )
          {
                printf("ArmPlugin - failed to find Gripper '%s'\n", GRIP_NAME);
                return;
          }
          
          // get the bounding box for the gripper
          const math::Box& gripBBox = gripper->GetBoundingBox();
          const float groundContact = 0.05f;
          
          if( gripBBox.min.z <= groundContact || gripBBox.max.z <= groundContact )
          {
                // set appropriate Reward for robot hitting the ground.
                printf("GROUND CONTACT, EOE\n");
                rewardHistory = REWARD_LOSS;
                newReward     = true;
                endEpisode    = true;
          }
          else
          {
                // Issue an interim reward based on the distance to the object
                const float distGoal = BoxDistance(gripBBox, propBBox); // compute the reward from distance to the goal
                
                if(DEBUG){printf("distance('%s', '%s') = %f\n", gripper->GetName().c_str(), prop->model->GetName().c_str(), distGoal);}
                
                // issue an interim reward based on the delta of the distance to the object
                if( episodeFrames > 1 )
                {
                    const float distDelta  = lastGoalDistance - distGoal;
                    const float movingAvg  = 0.95f;
                    const float timePenalty  =  0.25f;
                    
                    // compute the smoothed moving average of the delta of the distance to the goal
                    avgGoalDelta  = (avgGoalDelta * movingAvg) + (distDelta * (1.0f - movingAvg));
                    rewardHistory = (avgGoalDelta - timePenalty) * REWARD_MULTIPLIER;
                    newReward     = true;
                }
                
                lastGoalDistance = distGoal;
          }
    }

Bounding box of the gripper determines whether it is contacted to the
gound or not. The distance to the tube was calcaulated into reward with
a moving average of 20 fraction. The time penalty was set to 0.25 to
penalise timeout status. With low or no penalty or interim reward, agent
may enjoy time lifting arms for no feedback or bending backwards and
going far from the tube. Without this intrim reward, the agent might
discard all the effort, bending joint towards the tube little by little,
and would not learn quickly. The deep network will not converge soon.

Hyperparameters
===============

Default Hyperparameters
-----------------------

The default parameters are summerised on table \[table:Default
Hyperparameters\]. These parameters are not changed from the default
settings.

       Parameter        Value
  -------------------- -------
   `VELOCITY_CONTRO`L   false
     `VELOCITY_MIN`     -0.2f
     `VELOCITY_MAX`     0.2f
    `INPUT_CHANNELS`      3
     `ALLOW_RANDOM`     true
      `DEBUG_DQN`       true
        `GAMMA`         0.9f
      `EPS_START`       0.9f
       `EPS_END`        0.05f
      `EPS_DECAY`        200

  : Summary of Default
  Hyperparameters[]{data-label="table:Default Hyperparameters"}

Custom Hyperparameters
----------------------

The customisation of parameters were inevitable to reflect the
environmental factors for the training. The gist of the defined
parameters are shown on Table \[table:Custom Hyperparameters\].
`INPUT_WIDTH` and `INPUT_HEIGHT` are a downsized from the subscribed
camera image feed. Usual RL Hyperparameters were defined, RMSProp
optimise, 0.1 of rather high learning rate for simple tasks, smaller
batch size for less memory space and affordable size. Replay memory is a
new concept for DQN and is set to 20000, which is big but this samples
with low coherency between samples and converges faster and more stable
during the training. Because of the training is timely expensive and
sometimes does not finish for a long time. Replay memory acts like
shadow training for boxers and treats experience data from the training
more valable. If the number of experience replay is small, the training
can be stuck in a local minima, with a large number of experience sets,
this does not makes any issues. LSTM was used to track the movement and
effectively makes the sequence labled and turn the training with single
image into a learning in consideration of sequences and occurences
together. In other words, the agents can see the casuality between time
frame and choose the best action instead choosing the action from a
snapshot of the time frame. LSTM size more than 256 was ineffective for
a short period of training like this experiment.

      Parameter       Value
  ----------------- ---------
    `INPUT_WIDTH`      64
   `INPUT_HEIGHT`      64
     `OPTIMIZER`     RMSprop
   `LEARNING_RATE`    0.1f
    `BATCH_SIZE`       32
   `REPLAY_MEMORY`    20000
     `USE_LSTM`       true
     `LSTM_SIZE`       256

  : Summary of Custom
  Hyperparameters[]{data-label="table:Custom Hyperparameters"}

Reward for winning was set higher than losing. Higher winning ratio
tends to converge the training faster and does not let the agent perform
unexpected actions avoiding not to get punished. Robot arm tends to
touch the object rather than trying not approach any part of the robot
arm to the ground. The Hyperparameters were used to initialise a DQN
agent in the function `ArmPlugin::createAgent` as shown in the listing
\[list:ArmPlugin::createAgent\].

    bool ArmPlugin::createAgent()
    {
          if( agent != NULL )
                return true;
          
          // Create DQN Agent
          agent = dqnAgent::Create(INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS, DOF*3, OPTIMIZER, LEARNING_RATE, REPLAY_MEMORY, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY,USE_LSTM, LSTM_SIZE, ALLOW_RANDOM, DEBUG_DQN);
          
          if( !agent )
          {
                printf("ArmPlugin - failed to create DQN agent\n");
                return false;
          }
               
          inputState = Tensor::Alloc(INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS);
          
          if( !inputState )
          {
                printf("ArmPlugin - failed to allocate %ux%ux%u Tensor\n", INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS);
                return false;
          }
          
          return true;
    }

`DOF*2` was used for `number of actions` parameter of the
`ArmPlugin::createAgent` function as there are three possible actions,
+, -.

Results
=======

Result of the experiment shows the agent is learning the task, touching
the gripper to the tube, were successfully performed.

Acheivements
------------

The first overall 80 percent of touching the tube with gripper is shown
on Figure \[fig:over80pRLARM\].

![Agent reaching overall 80 percent
accuracy[]{data-label="fig:over80pRLARM"}](./report/img/over80pRLARM.png)

200 iterations more the average 90 percent of contact to the tube could
be realised, Figure \[fig:over90p\], this is above the project
requirement. After this point agent’s `EPS_END` is 0.05 and rarely
starts new exploration in control. Therefore, more than 95 percent of
gripper touching the tube could be realised.

![Agent reaching overall 90 percent
accuracy[]{data-label="fig:over90p"}](./report/img/over90p.png)

Actual moment of touching the gripper to the tube is snapshoted and
shown on Figure \[fig:touchingGR86p\].

![Agent reaching reaching with
gripper[]{data-label="fig:touchingGR86p"}](./report/img/touchingGR86p.png)

The videos realated to those images are present in `./Videos` folder of
the project. `./Videos/BeginToLearn.avi` shows struggling agent not
knowing what is an expected action. Other videos shows an experienced
agens touching the tube confididently, knowing what are actions with
rewards over many episodes of learning.

Discussion
==========

This experiment shows usage of premade example of jetson reinforcement
repository. The building of the libraries of torch was too hard on
jetson Xavier, it had to be tested on the udacity workspace, however,
some build errors lurks up and had make the environment for the
workspace using `sudo apt-get install libignition-math2-dev` to build
the `ArmPlugin.cpp`. This repository should be updated for new
architectures and prefer not to use a library not readily accessible,
pytorch. The backend of the DQN RL could be with tensorflow RT instead
of importing python object processed by pytorch, which can be a issue
with realtime robot cases by injecting performance issues. If saving DQN
weights function was implemented it could be better to continue the
training after pausing the simulation. Abandonning trained agents’
weight every time disconnecting from the workspace seem to be a waste of
computing resources and processing time.

Conclusion / Future work
========================

The more advanced reinforcement agent algirithms are available. Proximal
Policy Optimization Algorithms(PPO) [@Schulman2017] is excelent with
continous state space and control space like drone flight controller or
rocket propulsion control with regard to the continous environmental
states. OoenAI foundation suggests Deep deterministic policy gradient
(DDPG)[@Lillicrap2015] and Hindsight Experience
Replay[@Andrychowicz2017],a new experience replay that can learn from
failure, as the top of the edge reinforcement learning algorithms and
proves their algorithms are outperforming conventional algorithms
easily. It is a rapidly developing area of research field. The
technology used in this project can be substituted with new algorithms
on cutting edge trends.


-----------------------------------------------

# Deep RL Arm Manipulation


This project is based on the Nvidia open source project "jetson-reinforcement" developed by [Dustin Franklin](https://github.com/dusty-nv). The goal of the project is to create a DQN agent and define reward functions to teach a robotic arm to carry out two primary objectives:

1. Have any part of the robot arm touch the object of interest, with at least a 90% accuracy.
2. Have only the gripper base of the robot arm touch the object, with at least a 80% accuracy.

## Building from Source (Nvidia Jetson TX2)

Run the following commands from terminal to build the project from source:

``` bash
$ sudo apt-get install cmake
$ git clone http://github.com/udacity/RoboND-DeepRL-Project
$ cd RoboND-DeepRL-Project
$ git submodule update --init
$ mkdir build
$ cd build
$ cmake ../
$ make
```

During the `cmake` step, Torch will be installed so it can take awhile. It will download packages and ask you for your `sudo` password during the install.

## Testing the API

To make sure that the reinforcement learners are still functioning properly from C++, a simple example of using the API called [`catch`](samples/catch/catch.cpp) is provided.  Similar in concept to pong, a ball drops from the top of the screen which the agent must catch before the ball reaches the bottom of the screen, by moving it's paddle left or right.

To test the textual [`catch`](samples/catch/catch.cpp) sample, run the following executable from the terminal.  After around 100 episodes or so, the agent should start winning the episodes nearly 100% of the time:  

``` bash
$ cd RoboND-DeepRL-Project/build/aarch64/bin
$ ./catch 
[deepRL]  input_width:    64
[deepRL]  input_height:   64
[deepRL]  input_channels: 1
[deepRL]  num_actions:    3
[deepRL]  optimizer:      RMSprop
[deepRL]  learning rate:  0.01
[deepRL]  replay_memory:  10000
[deepRL]  batch_size:     32
[deepRL]  gamma:          0.9
[deepRL]  epsilon_start:  0.9
[deepRL]  epsilon_end:    0.05
[deepRL]  epsilon_decay:  200.0
[deepRL]  allow_random:   1
[deepRL]  debug_mode:     0
[deepRL]  creating DQN model instance
[deepRL]  DQN model instance created
[deepRL]  DQN script done init
[cuda]  cudaAllocMapped 16384 bytes, CPU 0x1020a800000 GPU 0x1020a800000
[deepRL]  pyTorch THCState  0x0318D490
[deepRL]  nn.Conv2d() output size = 800
WON! episode 1
001 for 001  (1.0000)  
WON! episode 5
004 for 005  (0.8000)  
WON! episode 10
007 for 010  (0.7000)  
WON! episode 15
010 for 015  (0.6667)  
WON! episode 20
013 for 020  (0.6500)  13 of last 20  (0.65)  (max=0.65)
WON! episode 25
015 for 025  (0.6000)  11 of last 20  (0.55)  (max=0.65)
LOST episode 30
018 for 030  (0.6000)  11 of last 20  (0.55)  (max=0.65)
LOST episode 35
019 for 035  (0.5429)  09 of last 20  (0.45)  (max=0.65)
WON! episode 40
022 for 040  (0.5500)  09 of last 20  (0.45)  (max=0.65)
LOST episode 45
024 for 045  (0.5333)  09 of last 20  (0.45)  (max=0.65)
WON! episode 50
027 for 050  (0.5400)  09 of last 20  (0.45)  (max=0.65)
WON! episode 55
031 for 055  (0.5636)  12 of last 20  (0.60)  (max=0.65)
LOST episode 60
034 for 060  (0.5667)  12 of last 20  (0.60)  (max=0.65)
WON! episode 65
038 for 065  (0.5846)  14 of last 20  (0.70)  (max=0.70)
WON! episode 70
042 for 070  (0.6000)  15 of last 20  (0.75)  (max=0.75)
LOST episode 75
045 for 075  (0.6000)  14 of last 20  (0.70)  (max=0.75)
WON! episode 80
050 for 080  (0.6250)  16 of last 20  (0.80)  (max=0.80)
WON! episode 85
055 for 085  (0.6471)  17 of last 20  (0.85)  (max=0.85)
WON! episode 90
059 for 090  (0.6556)  17 of last 20  (0.85)  (max=0.85)
WON! episode 95
063 for 095  (0.6632)  18 of last 20  (0.90)  (max=0.90)
WON! episode 100
068 for 100  (0.6800)  18 of last 20  (0.90)  (max=0.90)
WON! episode 105
073 for 105  (0.6952)  18 of last 20  (0.90)  (max=0.90)
WON! episode 110
078 for 110  (0.7091)  19 of last 20  (0.95)  (max=0.95)
WON! episode 111
079 for 111  (0.7117)  19 of last 20  (0.95)  (max=0.95)
WON! episode 112
080 for 112  (0.7143)  20 of last 20  (1.00)  (max=1.00)
```

Internally, [`catch`](samples/catch/catch.cpp) is using the [`dqnAgent`](c/dqnAgent.h) API from our C++ library to implement the learning.


## Project Environment

To get started with the project environment, run the following:

``` bash
$ cd RoboND-DeepRL-Project/build/aarch64/bin
$ chmod u+x gazebo-arm.sh
$ ./gazebo-arm.sh
```

<img src="https://github.com/dusty-nv/jetson-reinforcement/raw/master/docs/images/gazebo_arm.jpg">

The plugins which hook the learning into the simulation are located in the `gazebo/` directory of the repo. The RL agent and the reward functions are to be defined in [`ArmPlugin.cpp`](gazebo/ArmPlugin.cpp).
