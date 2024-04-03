import os
import time

import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

from render_tools import *
from math_tools import *

import pypbd as pbd
import numpy as np

# 1 = distance constraints (PBD)
# 2 = FEM triangle constraints (PBD)
# 3 = strain triangle constraints (PBD)
# 4 = distance constraints (XPBD)
simModel = 4

# 1 = dihedral angle (PBD)
# 2 = isometric bending (PBD)
# 3 = isometric bending  (XPBD)   
bendingModel = 3

# Get the absolute path of the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change the working directory to the script's directory
os.chdir(script_dir)

def buildModel():
    sim = pbd.Simulation.getCurrent()
    
    # Init simulation and collision detection with standard settings
    sim.initDefault()
    model = sim.getModel()
    rbs = model.getRigidBodies()
    
    # Create the mesh for the cloth model
    createMesh(simModel, bendingModel, "models/shirts.obj", translation=[0, -0.2, 0], scale=4.0, thickness=0.0)
    createMesh(simModel, bendingModel, "models/vest.obj", translation=[0, -0.2, 0], scale=4.0, thickness=0.05)
    createMeshAlt(simModel, bendingModel, translation=[0, 0, 0], scale=1.0, thickness=0.0)
    
    model_path = "smplx/SMPLX_NEUTRAL.npz"
    # shape = [0,0,0,0,0,0,0,0,0,0]
    shape = [2, 0, -.91, 0.082, 1, -2, .832, -0.106, 0.644, 0.548]
    sdf_resolution = [20, 20, 20]
    scale = 2.0
    min_extent_pad = 0.0 # i would start with a padding of 2.0 with resolution 30 
    max_extent_pad = 0.0

    global actor_rigidbody
    (actor_rigidbody) = model.setActor(model_path, shape, sdf_resolution, scale, [min_extent_pad, max_extent_pad])

    rest_pose = [0.0] * 66
    rest_pose[3*16+2] = -0.333
    rest_pose[3*17+2] = 0.333
    model.setActorPose(rest_pose)
    
    model.setContactStiffnessParticleRigidBody(100.0)
    
    # Set collision tolerance (distance threshold)
    cd = sim.getTimeStep().getCollisionDetection()      
    cd.setTolerance(0.025)
    
    # Set the number of substeps for a timestep
    ts = sim.getTimeStep()
    ts.setValueUInt(pbd.TimeStepController.NUM_SUB_STEPS, 8)

     
# Create a particle model mesh 
def createMesh(simModel, bendingModel, fileName, translation = [0, 0, 0], scale = 1.0, thickness = 0.0):
    sim = pbd.Simulation.getCurrent()
    model = sim.getModel()
    
    # Load geometry from file
    (vd, mesh) = pbd.OBJLoader.loadObjToMesh(fileName, [scale, scale, scale])

    # move up
    vertices = np.array(vd.getVertices())
    for x in vertices:
        x[0] += translation[0]
        x[1] += translation[1]
        x[2] += translation[2]
    triModel = model.addTriangleModel(vertices, mesh.getFaces(), testMesh=True)
    triModel.setThickness(thickness)
    triModel.setFrictionCoeff(0.9)

    # init constraints
    stiffness = 1.0
    if (simModel == 4):
        stiffness = 100000
    poissonRatio = 0.3
    
    # Generate a default set of constraints for the cloth model to simulate stretching and shearing resistance
    # simModel: 
    # 1 = distance constraints (PBD)
    # 2 = FEM triangle constraints (PBD)
    # 3 = strain triangle constraints (PBD)
    # 4 = distance constraints (XPBD)
    model.addClothConstraints(triModel, simModel, stiffness, stiffness, stiffness, stiffness, 
        poissonRatio, poissonRatio, False, False)

    bending_stiffness = 0.5
    if (bendingModel == 3):
        bending_stiffness = 10.0
        
    # Generate a default set of bending constraints for the cloth model
    # bendingModel:
    # 1 = dihedral angle (PBD)
    # 2 = isometric bending (PBD)
    # 3 = isometric bending  (XPBD)   
    model.addBendingConstraints(triModel, bendingModel, bending_stiffness)

# Create a particle model mesh with numpy array
def createMeshAlt(simModel, bendingModel, translation = [0, 0, 0], scale = 1.0, thickness = 0.0):
    sim = pbd.Simulation.getCurrent()
    model = sim.getModel()
    
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0]
    ])

    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [2, 3, 7],
        [2, 7, 6],
        [0, 3, 7],
        [0, 7, 4],
        [1, 2, 6],
        [1, 6, 5]
    ])

    (vd, mesh) = model.createMesh(vertices, faces, scale / 2)

    # move up
    vertices = np.array(vd.getVertices())
    for x in vertices:
        x[0] += translation[0]
        x[1] += translation[1]
        x[2] += translation[2]
    triModel = model.addTriangleModel(vertices, mesh.getFaces(), testMesh=True)
    triModel.setThickness(thickness)
    triModel.setFrictionCoeff(0.9)

    # init constraints
    stiffness = 1.0
    if (simModel == 4):
        stiffness = 100000
    poissonRatio = 0.3
    
    model.addClothConstraints(triModel, simModel, stiffness, stiffness, stiffness, stiffness, 
        poissonRatio, poissonRatio, False, False)

    bending_stiffness = 0.5
    if (bendingModel == 3):
        bending_stiffness = 10.0
  
    model.addBendingConstraints(triModel, bendingModel, bending_stiffness)

# Render all bodies            
def render():
    sim = pbd.Simulation.getCurrent()
    model = sim.getModel()
    
    # render meshes of rigid bodies
    rbs = model.getRigidBodies() 
    for rb in rbs:
        vd = rb.getGeometry().getVertexData()
        # you can access raw vertex data and mesh data
        # np.array(vd.getVertices())
        # np.array(mesh.getFaces())
        mesh = rb.getGeometry().getMesh()
        drawMesh(vd, mesh, 0, [0,0.2,0.7])
    
    colors = [
        [0.8, 0.9, 0.2, 1],
        [0.2, 0.8, 0.9, 1],
        [0.9, 0.2, 0.8, 1],
        [0.8, 0.2, 0.9, 1]
    ]

    # render meshes of cloth models
    triModels = model.getTriangleModels()
    for i, triModel in enumerate(triModels):
        pd = model.getParticles()
        offset = triModel.getIndexOffset()
        drawMesh(pd, triModel.getParticleMesh(), offset, colors[(i + len(rbs)) % len(colors)])


    # render time    
    glPushMatrix()
    glLoadIdentity()
    drawText([-0.95,0.9], "Time: {:.2f}".format(pbd.TimeManager.getCurrent().getTime()))
    glPopMatrix()
    
# Perform simulation steps
def timeStep():
    sim = pbd.Simulation.getCurrent()
    model = sim.getModel()
    
    current_time = pbd.TimeManager.getCurrent().getTime()
    wave = math.sin(current_time) / 8
    
    pose = [wave] * 66
    pose[3*16+2] = -0.333 + wave
    pose[3*17+2] = 0.333 + wave
    
    model.setActorPose(pose)

    global actor_rigidbody
    (dist, normal) = model.querySDF(actor_rigidbody, [2,0,0])
    (dist_list, normal_list) = model.querySDFBatch(actor_rigidbody, [[2,0,0],[2,1,0]])

    # # bench
    # positions = np.array([[2 * i / 999.0 - 7, 2 * (i / 999.0 + 0.33) - 5, 2 * (i / 999.0 + 0.67) - 6] for i in range(1000)])

    # start1 = time.time()
    # for position in positions:
    #     dist, normal = model.querySDF(actor_rigidbody, position)
    # end1 = time.time()
    # diff1 = (end1 - start1) * 1000

    # start2 = time.time()
    # dist_list, normal_list = model.querySDFBatch(actor_rigidbody, positions)
    # end2 = time.time()
    # diff2 = (end2 - start2) * 1000

    # print(f"Time taken for 1000 single queries: {diff1:.7f} ms")
    # print(f"Time taken for one batch query of size 1000: {diff2:.7f} ms")
    
    # Here set the number of timesteps per frame
    for i in range(8):
        sim.getTimeStep().step(model)
        
    # Update mesh normals of cloth model for rendering
    for triModel in model.getTriangleModels():
        triModel.updateMeshNormals(model.getParticles())
     
# Reset the simulation
def reset():
    sim = pbd.Simulation.getCurrent()
    sim.reset()
    sim.getModel().cleanup()
    cd = sim.getTimeStep().getCollisionDetection()      
    cd.cleanup()
    buildModel()
    
def main():
    # Activate logger to output info on the console
    pbd.Logger.addConsoleSink(pbd.LogLevel.INFO)

    # init OpenGL
    initGL(640, 720)  
    gluLookAt(0, 2.5, 5, 0, 0, 0, 0, 1, 0)
    # glRotatef(20.0, 0, 1, 0)
    
    # build simulation model 
    buildModel()

    # start event loop 
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    reset()
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glClearColor(0.3, 0.3, 0.3, 1.0)
        timeStep()
        render()
        pygame.display.flip()
        
    pygame.quit()
    pbd.Timing.printAverageTimes()

if __name__ == "__main__":
    main()