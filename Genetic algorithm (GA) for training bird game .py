"""
******************************

important peices of code are commented with #important, the rest is just code to run the game

References:
The Base game components was taken from - https://github.com/techwithtim/NEAT-Flappy-Bird/commit/e80296cc5feb1f0911c9066ace352a7c8e6dd0c3
Code to intergrate the base game and neat algorithm was taken from - https://github.com/techwithtim/NEAT-Flappy-Bird/blob/master/flappy_bird.py
Referenced youtube playlist - https://www.youtube.com/watch?v=OGHA-elMrxI&ab_channel=TechWithTim


Changes from orginal reference material:
- changed input nodes from bird height, distance to top pipe, distance to bottom pipe to bird height, top pipe height and bottom pipe height increasing the required complexity of the neural netwrok
- changes config file and algorithm settings
    - used hidden layer nodes reference material did not use them
    - removed bias nodes
    - removed node response weights
    - changed node weight design
    - changed output threshold function from tanh to y=x
    - changed how fitness is processed
    - simplifed the overall process to increase the amount of work needed to be done by the neural network
********************************
"""
# important: libaries needed to be imported

import pygame  # used to run game engine
import random  # used to randomly generate pipe heights for the game
import os  # used to access config file
import neat  # neat is the genetic/neural algorithm used

"""
******************************************************************************************
Start of Base game code from github, No changes made here from the orginal 
******************************************************************************************
"""
pygame.font.init()  # init font
WIN_WIDTH = 600
WIN_HEIGHT = 800
FLOOR = 730
STAT_FONT = pygame.font.SysFont("comicsans", 50)
END_FONT = pygame.font.SysFont("comicsans", 70)
DRAW_LINES = False
WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Flappy Bird")
pipe_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","pipe.png")).convert_alpha())
bg_img = pygame.transform.scale(pygame.image.load(os.path.join("imgs","bg.png")).convert_alpha(), (600, 900))
bird_images = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird" + str(x) + ".png"))) for x in range(1,4)]
base_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","base.png")).convert_alpha())
generation = 0


class Bird:
    MAX_ROTATION = 25
    IMGS = bird_images
    ROT_VEL = 20
    ANIMATION_TIME = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0  # degrees to tilt
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]

    def jump(self):
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count += 1
        displacement = self.vel*(self.tick_count) + 0.5*(3)*(self.tick_count)**2  # calculate displacement
        if displacement >= 16:
            displacement = (displacement/abs(displacement)) * 16
        if displacement < 0:
            displacement -= 2
        self.y = self.y + displacement
        if displacement < 0 or self.y < self.height + 50:  # tilt up
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL

    def draw(self, win):
        self.img_count += 1
        if self.img_count <= self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count <= self.ANIMATION_TIME*2:
            self.img = self.IMGS[1]
        elif self.img_count <= self.ANIMATION_TIME*3:
            self.img = self.IMGS[2]
        elif self.img_count <= self.ANIMATION_TIME*4:
            self.img = self.IMGS[1]
        elif self.img_count == self.ANIMATION_TIME*4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0
        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME*2
        blitRotateCenter(win, self.img, (self.x, self.y), self.tilt)

    def get_mask(self):
        return pygame.mask.from_surface(self.img)


class Pipe():
    GAP = 200
    VEL = 5

    def __init__(self, x):
        self.x = x
        self.height = 0
        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(pipe_img, False, True)
        self.PIPE_BOTTOM = pipe_img
        self.passed = False
        self.set_height()

    def set_height(self):
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        self.x -= self.VEL

    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird, win):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)
        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))
        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask,top_offset)
        if b_point or t_point:
            return True
        return False


class Base:
    VEL = 5
    WIDTH = base_img.get_width()
    IMG = base_img

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL
        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))


def blitRotateCenter(surf, image, topleft, angle):
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center = image.get_rect(topleft = topleft).center)
    surf.blit(rotated_image, new_rect.topleft)


def draw_window(win, birds, pipes, base, score, generation, pipe_ind):
    if generation == 0:
        generation = 1
    win.blit(bg_img, (0,0))
    for pipe in pipes:
        pipe.draw(win)
    base.draw(win)
    for bird in birds:
        if DRAW_LINES:
            try:
                pygame.draw.line(win, (255,0,0), (bird.x+bird.img.get_width()/2, bird.y + bird.img.get_height()/2), (pipes[pipe_ind].x + pipes[pipe_ind].PIPE_TOP.get_width()/2, pipes[pipe_ind].height), 5)
                pygame.draw.line(win, (255,0,0), (bird.x+bird.img.get_width()/2, bird.y + bird.img.get_height()/2), (pipes[pipe_ind].x + pipes[pipe_ind].PIPE_BOTTOM.get_width()/2, pipes[pipe_ind].bottom), 5)
            except:
                pass
        bird.draw(win)
    score_label = STAT_FONT.render("Score: " + str(score),1,(255,255,255))
    win.blit(score_label, (WIN_WIDTH - score_label.get_width() - 15, 10))
    score_label = STAT_FONT.render("Generation: " + str(generation-1),1,(255,255,255))
    win.blit(score_label, (10, 10))
    score_label = STAT_FONT.render("Alive: " + str(len(birds)),1,(255,255,255))
    win.blit(score_label, (10, 50))
    pygame.display.update()



"""
******************************************************************************************
Running the game and determining the fitness of the species, most code is the same as the orginal expect inputs for the neural network and the fitness function  
******************************************************************************************
"""

def evaluate_fitness_of_Population(genomes, config):

    global WIN, generation
    win = WIN #game display window
    generation += 1 #generation / iteration

    neural_network_list = []
    birds = []
    genome_list = []


    for genome_id, genome in genomes:  # For each bird 'genome' in the population 'genomes'
        genome.fitness = 0 #set inital fitness to zero
        neural_network = neat.nn.FeedForwardNetwork.create(genome, config) #create the neural network based on the genome
        neural_network_list.append(neural_network) #add the neural network to the neural network list
        birds.append(Bird(230,350)) #used to set deafult postion of the bird
        genome_list.append(genome) #add the genome to the genome list


    base = Base(FLOOR)
    pipes = [Pipe(700)]
    score = 0
    clock = pygame.time.Clock()

    run = True
    while run and len(birds) > 0:
        clock.tick(40) #speed
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
                break

        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1

        for x, bird in enumerate(birds):
            genome_list[x].fitness += 0.01 # add a partial fitness value for surving each frame, reducing ability for birds that die right move into next generation
            bird_height = bird.y
            top_pipe_height = pipes[pipe_ind].height
            bottom_pipe_height = pipes[pipe_ind].bottom
            bird.move()

            # important: get the output node value by giving the inputs and getting the neural network to give the ouput value
            output_node_value = neural_network_list[birds.index(bird)].activate((bird_height, top_pipe_height, bottom_pipe_height))

            #important: if the output node value is greater than one jump
            # the basis for this is that the neural network should ideally calcualte the difference between the centre height between the pipes and the bird (centre - bird height = difference)
            if output_node_value[0] >= 1: # if the difference is greater than one aka the bird is below the centre line, jump!
                bird.jump()

        base.move()

        rem = []
        add_pipe = False
        for pipe in pipes:
            pipe.move()

            for bird in birds:
                if pipe.collide(bird, win):
                    neural_network_list.pop(birds.index(bird))
                    birds.pop(birds.index(bird))

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            if not pipe.passed and pipe.x < bird.x:
                pipe.passed = True
                add_pipe = True

        #important: fitness calculation
        if add_pipe:
            score += 1
            for genome in genome_list:
                genome.fitness += 1 #if a bird succesfully passes through a pipe, increase its fitness by one
            pipes.append(Pipe(WIN_WIDTH))

        for r in rem:
            pipes.remove(r)

        for bird in birds:
            if bird.y + bird.img.get_height() - 10 >= FLOOR or bird.y < -50:
                neural_network_list.pop(birds.index(bird))
                genome_list.pop(birds.index(bird))
                birds.pop(birds.index(bird))

        draw_window(WIN, birds, pipes, base, score, generation, pipe_ind)


def run(config_file):
    # important
    # load the neat config file that contains date like population size, mutation rate, node generation rate and etc
    # neat config function parses based on setting groups like genome, species and reproduction
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)

    """
    #important 
    ******************************
    Config file Contents as a comment: 
    ********************************

    [NEAT]
    #Basic settings 
    fitness_criterion     = max  #this is set to max as we want select for birds that achieve the highest score, the min option will select for the lowest score  
    fitness_threshold     = 20 #if a bird can get a score of 20, we can safely assume it will be able to achieve an even higher score like 100 or 200 without any issues. However this value is arbitrary and is used to reduce computing time 
    pop_size              = 100  
    reset_on_extinction   = False #if stagnation occurs (no increase in fitness over x iterations) end the program 

    [DefaultGenome]
    #output layer node function 
    activation_default      = identity #the identity function is (y=x), this suitable for the code as the bird will only jump if the difference is greater than a specific number, we don't need any complicated functions like tanx for our application 
    activation_mutate_rate  = 0.0 #we don't need to test out any different activation functions, so we will set the mutation to zero 
    activation_options      = identity #if we wanted to try different activation functions we would add them here 

    # hidden layer node function options
    aggregation_default     = sum #the default node will be the add the inputs together 
    aggregation_mutate_rate = 0.1 #this the mutation rate that the node function will change 
    aggregation_options     = sum, mean # we will also add the option of mean (this is useful when we want to calculate the centre height between the two pipes), there is no difference option 

    # Bias node options
    # we don't need bias nodes for our application so this section can be ignored 
    bias_init_mean          = 0.0
    bias_init_stdev         = 1.0
    bias_max_value          = 0 #setting the min and max to zero will ensure the bias nodes won't have any impact. Bias node functionality can't be turned off so we have to do this 
    bias_min_value          = 0 
    bias_mutate_power       = 0.5
    bias_mutate_rate        = 0.7
    bias_replace_rate       = 0.1

    # genome compatibility options
    #this section will determine when the software will consider two models to be from the same species or two different species 
    #I don't understand what they mean myself so I can't explain what they do 
    compatibility_disjoint_coefficient = 1.0  
    compatibility_weight_coefficient   = 0.5

    # connection add/remove rates
    conn_add_prob           = 0.5 #50% chance to add a connection to a node 
    conn_delete_prob        = 0.5 #50% chance to remove a connection to a node 

    # connection enable options
    enabled_default         = True #enable ("turn on") connections by default
    enabled_mutate_rate     = 0.01  #1% chance to disable ("Turn off") a connection 

    feed_forward            = True #don't allow node outputs to become node inputs 
    initial_connection      = full #at the start all input nodes will connect to the hidden nodes 

    # node add/remove rates
    node_add_prob           = 0.2 #20% chance to add a hidden layer node
    node_delete_prob        = 0.2 #20% chance to remove a hidden layer node 

    # network parameters
    num_hidden              = 2  # we will only need one or two hidden layer nodes to calculate the centre height 
    num_inputs              = 3 # the 3 inputs will be birdHeight,topPipeHeight,BottomPipeHeight
    num_outputs             = 1 #the output will ideally be the difference between the centre of the two pipes and the bird height 

    # node response options
    # activation(bias+(response∗aggregation(inputs)))
    #not needed for our application so we can ingnore it 

    response_init_mean      = 1.0
    response_init_stdev     = 0.0
    response_max_value      = 1 #setting a min and max to one will ensure it has no impact. There is no option to turn off response functionality 
    response_min_value      = 1
    response_mutate_power   = 0.0
    response_mutate_rate    = 0.0
    response_replace_rate   = 0.0

    # connection weight options
    weight_init_mean        = 0.0 #relates to spread 
    weight_init_stdev       = 0.5 # having a mean of 0.5 will ensure we hit all of our needed weights -1,0.5 and 1. -0.5 is also a possibility but will not be helpful
    weight_max_value        = 1 #can correspond adding input value 
    weight_min_value        = -1 #can correspond to subtracting input value 
    weight_mutate_power     = 0.5 #50% chance the std deviation will added or subtracted from the connection weight 
    weight_mutate_rate      = 0.8 #80% chance of changing weight by adding a random value 
    weight_replace_rate     = 0.1 #10% change to replace the connection weight 

    [DefaultSpeciesSet]
    compatibility_threshold = 3.0

    [DefaultStagnation]
    species_fitness_func = max 
    max_stagnation       = 10 #if after 10 generations no increase in fitness occurs the species will go extinct 
    species_elitism      = 2 # at least 2 species will remain even if stagnation occurs 

    [DefaultReproduction]
    elitism            = 2 # 2 of the best performing species will be retained each generation 
    survival_threshold = 0.2 #only top 20% of the species can reproduce 
    """

    population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(False))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    smartest_Bird = population.run(evaluate_fitness_of_Population, 50) # the smartest bird is the bird with the highest score over 50 generations

    print('\nBest Bird:\n{!s}'.format(smartest_Bird))


if __name__ == '__main__':
    # function used to locate the confing file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)

