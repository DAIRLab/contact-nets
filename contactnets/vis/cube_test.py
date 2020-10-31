import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import glfw

import imageio

import pdb

light_diffuse = [1.0, 1.0, 1.0, 1.0]
light_position = [0.0, 0.0, 1.0, 1.0]
# Normals for the 6 faces of a cube
normals = [
    (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0), (0.0, 0.0, -1.0), (0.0, 0.0, 1.0)
]
# Vertex indices for the 6 faces of a cube
faces = [
    [0, 1, 2, 3], [3, 2, 6, 7], [7, 6, 5, 4],
    [4, 5, 1, 0], [5, 6, 2, 1], [7, 4, 0, 3]
];
vertices = [
    (-1.0, -1.0, 1.0), (-1.0, -1.0, -1.0), (-1.0, 1.0, -1.0), (-1.0, 1.0, 1.0),
    (1.0, -1.0, 1.0),  (1.0, -1.0, -1.0),  (1.0, 1.0, -1.0),  (1.0, 1.0, 1.0)
]

def draw_cube():
    for i in range(6):
        glBegin(GL_QUADS)

        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, [1.0, 0.0, 0.0, 1.0])

        glNormal3fv(normals[i])
        glVertex3fv(vertices[faces[i][0]])
        glVertex3fv(vertices[faces[i][1]])
        glVertex3fv(vertices[faces[i][2]])
        glVertex3fv(vertices[faces[i][3]])
        glEnd()

def main():
    DISPLAY_WIDTH = 900
    DISPLAY_HEIGHT = 900
    # Initialize the library
    if not glfw.init():
        return
    # Set window hint NOT visible
    # glfw.window_hint(glfw.VISIBLE, False)
    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(DISPLAY_WIDTH, DISPLAY_HEIGHT, "hidden window", None, None)
    if not window:
        glfw.terminate()
        return

    # Make the window's context current
    glfw.make_context_current(window)

    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
    glLightfv(GL_LIGHT0, GL_POSITION, light_position)
    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHTING)

    glEnable(GL_DEPTH_TEST)
    
    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (DISPLAY_WIDTH / DISPLAY_HEIGHT), 0.1, 100.0)

    glMatrixMode(GL_MODELVIEW)
    glTranslatef(0.0, 0.0, -5.0)

    for i in range(50):
        glPushMatrix()
        glRotatef(i * 5, 3, 1, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_cube()
        glPopMatrix()

        # image_buffer = glReadPixels(0, 0, DISPLAY_WIDTH, DISPLAY_HEIGHT, OpenGL.GL.GL_RGB, OpenGL.GL.GL_UNSIGNED_BYTE)
        # image = np.frombuffer(image_buffer, dtype=np.uint8).reshape(DISPLAY_WIDTH, DISPLAY_HEIGHT, 3)
        # imageio.imwrite(f'snaps/image{i}.png', image)
        glfw.swap_buffers(window)
    
    glfw.destroy_window(window)
    glfw.terminate()

if __name__ == "__main__":
    main()
