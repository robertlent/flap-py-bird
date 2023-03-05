import pygame
import neat
import time
import os
import random

pygame.font.init()

WIN_WIDTH = 500
WIN_HEIGHT = 800

GEN = 0

BIRD_IMAGES = [
    pygame.transform.scale2x(pygame.image.load(
        os.path.join("img", "bird1.png"))),
    pygame.transform.scale2x(pygame.image.load(
        os.path.join("img", "bird2.png"))),
    pygame.transform.scale2x(pygame.image.load(
        os.path.join("img", "bird3.png")))
]
PIPE_IMAGE = pygame.transform.scale2x(
    pygame.image.load(os.path.join("img", "pipe.png")))
BASE_IMAGE = pygame.transform.scale2x(
    pygame.image.load(os.path.join("img", "base.png")))
BG_IMAGE = pygame.transform.scale2x(
    pygame.image.load(os.path.join("img", "bg.png")))

STAT_FONT = pygame.font.SysFont("comicsan", 50)


class Bird:
    IMAGES = BIRD_IMAGES
    MAX_ROTATION = 25
    ROTATION_VELOCITY = 20
    ANIMATION_TIME = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.velocity = 0
        self.height = self.y
        self.image_count = 0
        self.image = self.IMAGES[0]

    def jump(self):
        self.velocity = -10.5
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count += 1

        displacement = self.velocity * self.tick_count + 1.5 * self.tick_count**2

        if displacement >= 16:
            displacement = 16

        if displacement < 0:
            displacement -= 2

        self.y = self.y + displacement

        if displacement < 0 or self.y < self.height + 50:
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:
            if self.tilt > -90:
                self.tilt -= self.ROTATION_VELOCITY

    def draw(self, surf):
        self.image_count += 1

        if self.image_count < self.ANIMATION_TIME:
            self.image = self.IMAGES[0]
        elif self.image_count < self.ANIMATION_TIME * 2:
            self.image = self.IMAGES[1]
        elif self.image_count < self.ANIMATION_TIME * 3:
            self.image = self.IMAGES[2]
        elif self.image_count < self.ANIMATION_TIME * 4:
            self.image = self.IMAGES[1]
        elif self.image_count == self.ANIMATION_TIME * 4 + 1:
            self.image = self.IMAGES[0]
            self.image_count = 0

        if self.tilt <= -80:
            self.image = self.IMAGES[1]
            self.image_count = self.ANIMATION_TIME * 2

        rotated_image = pygame.transform.rotate(self.image, self.tilt)
        new_rect = rotated_image.get_rect(
            center=self.image.get_rect(topleft=(self.x, self.y)).center)
        surf.blit(rotated_image, new_rect.topleft)

    def get_mask(self):
        return pygame.mask.from_surface(self.image)


class Pipe:
    GAP = 200
    VELOCITY = 5

    def __init__(self, x):
        self.x = x
        self.height = 0

        self.top = 0
        self.bottom = 0
        self.PIPE_BOTTOM = PIPE_IMAGE
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMAGE, False, True)

        self.passed = False
        self.set_height()

    def set_height(self):
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        self.x -= self.VELOCITY

    def draw(self, surf):
        surf.blit(self.PIPE_TOP, (self.x, self.top))
        surf.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        bottom_collision = bird_mask.overlap(bottom_mask, bottom_offset)
        top_collision = bird_mask.overlap(top_mask, top_offset)

        if top_collision or bottom_collision:
            return True

        return False


class Base:
    VELOCITY = 5
    WIDTH = BASE_IMAGE.get_width()
    IMAGE = BASE_IMAGE

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.VELOCITY
        self.x2 -= self.VELOCITY

        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, surf):
        surf.blit(self.IMAGE, (self.x1, self.y))
        surf.blit(self.IMAGE, (self.x2, self.y))


def draw_window(surf, birds, pipes, base, score, gen):
    surf.blit(BG_IMAGE, (0, 0))

    for pipe in pipes:
        pipe.draw(surf)

    text = STAT_FONT.render("Score: " + str(score), 1, "White")
    surf.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))

    text = STAT_FONT.render("Gen: " + str(gen), 1, "White")
    surf.blit(text, (10, 10))

    base.draw(surf)

    for bird in birds:
        bird.draw(surf)

    pygame.display.update()


def main(genomes, config):
    global GEN
    GEN += 1
    networks = []
    ge = []
    birds = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        networks.append(net)
        birds.append(Bird(230, 250))
        g.fitness = 0
        ge.append(g)

    base = Base(730)
    pipes = [Pipe(650)]
    surf = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()

    score = 0

    run = True

    while run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        pipe_index = 0

        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_index = 1
        else:
            run = False
            break

        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness += 0.1

            output = networks[x].activate((bird.y, abs(
                bird.y - pipes[pipe_index].height), abs(bird.y - pipes[pipe_index].bottom)))

            if output[0] > 0.5:
                bird.jump()

        add_pipe = False
        removed = []

        for pipe in pipes:
            for x, bird in enumerate(birds):
                if pipe.collide(bird):
                    ge[x].fitness -= 1
                    birds.pop(x)
                    networks.pop(x)
                    ge.pop(x)

                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                removed.append(pipe)

            pipe.move()

        if add_pipe:
            score += 1

            for g in ge:
                g.fitness += 5

            pipes.append(Pipe(650))

        for pipe in removed:
            pipes.remove(pipe)

        for x, bird in enumerate(birds):
            if bird.y + bird.image.get_height() >= 730 or bird.y < 0:
                birds.pop(x)
                networks.pop(x)
                ge.pop(x)

        if score > 50:
            break

        base.move()
        draw_window(surf, birds, pipes, base, score, GEN)


def run(config_path):
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                config_path)

    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    winner = pop.run(main, 50)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)
