import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import random

# ---------------- SETTINGS ----------------
WIDTH, HEIGHT = 800, 600
NUM_STARS = 1000
BLACK_HOLE_RADIUS = 1.0

# ---------------- INIT ----------------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
pygame.display.set_caption("Black Hole Simulator with Shaders")
gluPerspective(45, WIDTH / HEIGHT, 0.1, 100.0)
glTranslatef(0, 0, -10)
glEnable(GL_DEPTH_TEST)
glEnable(GL_POINT_SMOOTH)
glPointSize(2)

# ---------------- STARFIELD ----------------
stars = np.random.uniform(-50, 50, (NUM_STARS, 3))
stars[:, 2] -= 20  # move stars away

# ---------------- SHADER SOURCE ----------------
vertex_shader = """
#version 330 core
layout(location = 0) in vec3 position;
void main()
{
    gl_Position = gl_ModelViewProjectionMatrix * vec4(position,1.0);
    gl_PointSize = 2.0;
}
"""

fragment_shader = """
#version 330 core
out vec4 fragColor;
uniform vec3 bh_pos;
uniform float bh_radius;
void main()
{
    vec2 uv = gl_PointCoord - vec2(0.5);
    float dist = length(uv);
    float glow = smoothstep(bh_radius, bh_radius*0.7, dist);
    fragColor = vec4(vec3(glow),1.0);
}
"""

# ---------------- SHADER COMPILATION ----------------
def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(shader).decode())
    return shader

def create_shader_program(vs_src, fs_src):
    program = glCreateProgram()
    vs = compile_shader(vs_src, GL_VERTEX_SHADER)
    fs = compile_shader(fs_src, GL_FRAGMENT_SHADER)
    glAttachShader(program, vs)
    glAttachShader(program, fs)
    glLinkProgram(program)
    if glGetProgramiv(program, GL_LINK_STATUS) != GL_TRUE:
        raise RuntimeError(glGetProgramInfoLog(program))
    return program

shader_program = create_shader_program(vertex_shader, fragment_shader)
glUseProgram(shader_program)
bh_pos_loc = glGetUniformLocation(shader_program, "bh_pos")
bh_radius_loc = glGetUniformLocation(shader_program, "bh_radius")

# ---------------- PARTICLES ----------------
particles = []
for _ in range(500):
    r = random.uniform(2,5)
    theta = random.uniform(0, 2*np.pi)
    phi = random.uniform(0, np.pi)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    vx = -y*0.1
    vy = x*0.1
    vz = 0
    particles.append([x, y, z, vx, vy, vz])

# ---------------- FUNCTIONS ----------------
def draw_stars():
    glBegin(GL_POINTS)
    for s in stars:
        glVertex3f(s[0], s[1], s[2])
    glEnd()

def draw_black_hole():
    glColor3f(0,0,0)
    quad = gluNewQuadric()
    gluSphere(quad, BLACK_HOLE_RADIUS,64,64)

def draw_particles():
    glBegin(GL_POINTS)
    for p in particles:
        glColor3f(1.0,0.5,0.0)
        glVertex3f(p[0], p[1], p[2])
    glEnd()

def update_particles():
    for p in particles:
        x, y, z, vx, vy, vz = p
        dx, dy, dz = -x, -y, -z
        dist = np.sqrt(dx*dx + dy*dy + dz*dz)
        if dist < BLACK_HOLE_RADIUS:
            r = random.uniform(2,5)
            theta = random.uniform(0, 2*np.pi)
            phi = random.uniform(0, np.pi)
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            vx = -y*0.1
            vy = x*0.1
            vz = 0
            p[:] = [x,y,z,vx,vy,vz]
            continue
        force = 500 / (dist**2)
        ax, ay, az = dx/dist * force, dy/dist * force, dz/dist * force
        vx += ax*0.1; vy += ay*0.1; vz += az*0.1
        x += vx*0.1; y += vy*0.1; z += vz*0.1
        p[:] = [x,y,z,vx,vy,vz]

# ---------------- MAIN LOOP ----------------
clock = pygame.time.Clock()
rot_x, rot_y = 20,0
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
        elif event.type == pygame.MOUSEMOTION and pygame.mouse.get_pressed()[0]:
            dx, dy = event.rel
            rot_x += dy*0.2
            rot_y += dx*0.2

    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glTranslatef(0,0,-10)
    glRotatef(rot_x,1,0,0)
    glRotatef(rot_y,0,1,0)

    glUniform3f(bh_pos_loc, 0,0,0)
    glUniform1f(bh_radius_loc, BLACK_HOLE_RADIUS)

    draw_stars()
    draw_black_hole()
    draw_particles()
    update_particles()

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
