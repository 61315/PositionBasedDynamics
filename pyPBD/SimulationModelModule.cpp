#include "common.h"

#include <Simulation/Simulation.h>
#include <Simulation/SimulationModel.h>
#include <Simulation/CubicSDFCollisionDetection.h>
#include "Utils/Logger.h"
#include <pyPBD/bind_pointer_vector.h>

#include "smplx/smplx.hpp"
#include "smplx/util.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

Utilities::IndexedFaceMesh mesh_smpl_full;
PBD::VertexData vd_smpl_full;

Utilities::IndexedFaceMesh mesh_smpl_culled;
PBD::VertexData vd_smpl_culled;

std::unique_ptr<smplx::BodyX> smpl_body;
std::unique_ptr<smplx::ModelX> smpl_model;

std::vector<std::array<int, 3>> filtered_faces;
std::vector<int> filtered_face_indices;
std::vector<int> filtered_vertex_indices;

Eigen::Matrix<unsigned int, 3, 1> resolution_(20, 20, 20);
Real scale_;
std::pair<float, float>& extension_ = std::make_pair(0.0f, 0.0f);

// update vd with the full smpl mesh (run every frame)
void mesh_vertex_update(PBD::VertexData& vd, float scale) {
	const unsigned int nPoints = (unsigned int)smpl_model->n_verts();

	for (unsigned int i = 0; i < nPoints; i++) {
		vd.setPosition(i, smpl_body->verts().row(i).cast<Real>() * scale);
	}
}

// update vd with culled smpl mesh for collision (run every frame)
void mesh_vertex_update_cull(PBD::VertexData& vd, float scale) {
	const unsigned int nPoints = filtered_vertex_indices.size();

	for (unsigned int i = 0; i < nPoints; i++) {
		vd.setPosition(i, smpl_body->verts().row(filtered_vertex_indices[i]).cast<Real>() * scale);
	}
}

// init a full vd/mesh object (not needed)
void smpl_to_mesh(PBD::VertexData& vd, Utilities::IndexedFaceMesh& mesh, const Real& scale = 1.0) {
	mesh.release();

	const unsigned int nPoints = (unsigned int)smpl_model->n_verts();
	const unsigned int nFaces = (unsigned int)smpl_model->n_faces();

	mesh.initMesh(nPoints, nFaces * 2, nFaces);

	vd.reserve(nPoints);
	for (unsigned int i = 0; i < nPoints; i++) {
		vd.addVertex(smpl_body->verts().row(i).cast<Real>() * scale);
	}

	for (unsigned int i = 0; i < nFaces; i++) {
		int posIndices[3];
		int texIndices[3];
		for (int j = 0; j < 3; j++) {
			posIndices[j] = smpl_model->faces(i, j);
		}

		mesh.addFace(&posIndices[0]);
	}
	mesh.buildNeighbors();

	mesh.updateNormals(vd, 0);
	mesh.updateVertexNormals(vd);

	LOG_INFO << "smpl_to_mesh: Number of triangles: " << nFaces;
	LOG_INFO << "smpl_to_mesh: Number of vertices: " << nPoints;
}

bool valid(int vertexIndex) {
	float vertex[3];
	vertex[0] = smpl_body->verts().row(vertexIndex)[0];
	vertex[1] = smpl_body->verts().row(vertexIndex)[1];

	return vertex[0] < 0.65 && vertex[0] > -0.65 && vertex[1] < 0.24 && vertex[1] > -0.65;	// smplx
}

// init a culled full vd/mesh object
void smpl_to_mesh_torso(PBD::VertexData& vd, Utilities::IndexedFaceMesh& mesh,
						const Real& scale = 1.0) {
	mesh.release();

	const unsigned int nPoints = (unsigned int)smpl_model->n_verts();
	const unsigned int nFaces = (unsigned int)smpl_model->n_faces();

	for (unsigned int i = 0; i < nFaces; i++) {
		std::array<int, 3> face;
		face[0] = smpl_model->faces(i, 0);
		face[1] = smpl_model->faces(i, 1);
		face[2] = smpl_model->faces(i, 2);
		if (valid(face[0]) && valid(face[1]) && valid(face[2])) {
			filtered_faces.push_back(face);
			filtered_face_indices.push_back(i);
			filtered_vertex_indices.push_back(face[0]);
			filtered_vertex_indices.push_back(face[1]);
			filtered_vertex_indices.push_back(face[2]);
		}
	}

	std::sort(filtered_vertex_indices.begin(), filtered_vertex_indices.end());
	filtered_vertex_indices.erase(
		std::unique(filtered_vertex_indices.begin(), filtered_vertex_indices.end()),
		filtered_vertex_indices.end());

	const unsigned int nFilteredFaces = filtered_faces.size();
	const unsigned int nFilteredVertices = filtered_vertex_indices.size();

	mesh.initMesh(nFilteredVertices, nFilteredFaces * 2, nFilteredFaces);

	vd.reserve(nFilteredVertices);
	for (const auto& vertexIndex : filtered_vertex_indices) {
		vd.addVertex(smpl_body->verts().row(vertexIndex).cast<Real>() * scale);
	}

	std::unordered_map<int, int> vertexIndexMap;
	for (unsigned int i = 0; i < nFilteredVertices; i++) {
		vertexIndexMap[filtered_vertex_indices[i]] = i;
	}

	for (const auto& face : filtered_faces) {
		int posIndices[3] = {vertexIndexMap[face[0]], vertexIndexMap[face[1]],
							 vertexIndexMap[face[2]]};
		mesh.addFace(&posIndices[0]);
	}

	mesh.buildNeighbors();

	mesh.updateNormals(vd, 0);
	mesh.updateVertexNormals(vd);

	LOG_INFO << "smpl_to_mesh_torso: Number of triangles: " << nFilteredFaces;
	LOG_INFO << "smpl_to_mesh_torso: Number of vertices: " << nFilteredVertices;
}

PBD::CubicSDFCollisionDetection::GridPtr generateSDF(const std::vector<Vector3r> &vertices, const std::vector<unsigned int>& faces, const Eigen::Matrix<unsigned int, 3, 1>& resolution, const std::pair<float, float>& extension = std::make_pair(0.0f, 0.0f))
{
    const unsigned int nFaces = faces.size()/3;
#ifdef USE_DOUBLE
    Discregrid::TriangleMesh sdfMesh((vertices[0]).data(), faces.data(), vertices.size(), nFaces);
#else
    // if type is float, copy vector to double vector
    std::vector<double> doubleVec;
    doubleVec.resize(3 * vertices.size());
    for (unsigned int i = 0; i < vertices.size(); i++)
        for (unsigned int j = 0; j < 3; j++)
            doubleVec[3 * i + j] = vertices[i][j];
    Discregrid::TriangleMesh sdfMesh(&doubleVec[0], faces.data(), vertices.size(), nFaces);
#endif
    Discregrid::TriangleMeshDistance md(sdfMesh);
    Eigen::AlignedBox3d domain;
    for (auto const& x : sdfMesh.vertices())
    {
        domain.extend(x);
    }
    domain.max() += 0.1 * Eigen::Vector3d::Ones();
    domain.min() -= 0.1 * Eigen::Vector3d::Ones();

    domain.min() -= extension.first * Eigen::Vector3d::Ones();
 	domain.max() += extension.second * Eigen::Vector3d::Ones();

    std::cout << "Set SDF resolution: " << resolution[0] << ", " << resolution[1] << ", " << resolution[2] << std::endl;
    auto sdf = std::make_shared<PBD::CubicSDFCollisionDetection::Grid>(domain, std::array<unsigned int, 3>({ resolution[0], resolution[1], resolution[2] }));
    auto func = Discregrid::DiscreteGrid::ContinuousFunction{};
    func = [&md](Eigen::Vector3d const& xi) {return md.signed_distance(xi).distance; };
    std::cout << "Generate SDF\n";
    sdf->addFunction(func, true);
    return sdf;
}

void SimulationModelModule(py::module m_sub) 
{
    py::class_<PBD::TriangleModel>(m_sub, "TriangleModel")
        .def(py::init<>())
        .def("getParticleMesh", (const PBD::TriangleModel::ParticleMesh& (PBD::TriangleModel::*)()const)(&PBD::TriangleModel::getParticleMesh))
        .def("cleanupModel", &PBD::TriangleModel::cleanupModel)
        .def("getIndexOffset", &PBD::TriangleModel::getIndexOffset)
        .def("initMesh", &PBD::TriangleModel::initMesh)
        .def("updateMeshNormals", &PBD::TriangleModel::updateMeshNormals)
        .def("getRestitutionCoeff", &PBD::TriangleModel::getRestitutionCoeff)
        .def("setRestitutionCoeff", &PBD::TriangleModel::setRestitutionCoeff)
        .def("getFrictionCoeff", &PBD::TriangleModel::getFrictionCoeff)
        .def("setFrictionCoeff", &PBD::TriangleModel::setFrictionCoeff)
        .def("getThickness", &PBD::TriangleModel::getThickness)
        .def("setThickness", &PBD::TriangleModel::setThickness);

    py::class_<PBD::TetModel>(m_sub, "TetModel")
        .def(py::init<>())
        .def("getInitialX", &PBD::TetModel::getInitialX)
        .def("setInitialX", &PBD::TetModel::setInitialX)
        .def("getInitialR", &PBD::TetModel::getInitialR)
        .def("setInitialR", &PBD::TetModel::setInitialR)
        .def("getInitialScale", &PBD::TetModel::getInitialScale)
        .def("setInitialScale", &PBD::TetModel::setInitialScale)

        .def("getSurfaceMesh", &PBD::TetModel::getSurfaceMesh)
        .def("getVisVertices", &PBD::TetModel::getVisVertices)
        .def("getVisMesh", &PBD::TetModel::getVisMesh)
        .def("getParticleMesh", (const PBD::TetModel::ParticleMesh & (PBD::TetModel::*)()const)(&PBD::TetModel::getParticleMesh))
        .def("cleanupModel", &PBD::TetModel::cleanupModel)
        .def("getIndexOffset", &PBD::TetModel::getIndexOffset)
        .def("initMesh", &PBD::TetModel::initMesh)
        .def("updateMeshNormals", &PBD::TetModel::updateMeshNormals)
        .def("attachVisMesh", &PBD::TetModel::attachVisMesh)
        .def("updateVisMesh", &PBD::TetModel::updateVisMesh)
        .def("getRestitutionCoeff", &PBD::TetModel::getRestitutionCoeff)
        .def("setRestitutionCoeff", &PBD::TetModel::setRestitutionCoeff)
        .def("getFrictionCoeff", &PBD::TetModel::getFrictionCoeff)
        .def("setFrictionCoeff", &PBD::TetModel::setFrictionCoeff);
 
    py::bind_pointer_vector<std::vector<PBD::TriangleModel*>>(m_sub, "VecTriangleModels");
    py::bind_pointer_vector<std::vector<PBD::TetModel*>>(m_sub, "VecTetModels");
    py::bind_pointer_vector<std::vector<PBD::RigidBody*>>(m_sub, "VecRigidBodies");
    py::bind_vector<std::vector<PBD::Constraint*>>(m_sub, "VecConstraints");

    py::class_<PBD::SimulationModel, GenParam::ParameterObject>(m_sub, "SimulationModel")
        .def(py::init<>())
        .def("init", &PBD::SimulationModel::init)
        .def("reset", &PBD::SimulationModel::reset)
        .def("cleanup", &PBD::SimulationModel::cleanup)
        .def("resetContacts", &PBD::SimulationModel::resetContacts)
        .def("updateConstraints", &PBD::SimulationModel::updateConstraints)
        .def("initConstraintGroups", &PBD::SimulationModel::initConstraintGroups)
        .def("addTriangleModel", [](
            PBD::SimulationModel& model,
            std::vector<Vector3r>& points,
            std::vector<unsigned int>& indices,
            const PBD::TriangleModel::ParticleMesh::UVIndices& uvIndices,
            const PBD::TriangleModel::ParticleMesh::UVs& uvs,
            const bool testMesh)
            {
                auto& triModels = model.getTriangleModels();
                int i = triModels.size();
                model.addTriangleModel(points.size(), indices.size()/3, points.data(), indices.data(), uvIndices, uvs);
                if (testMesh)
                {
                    PBD::ParticleData& pd = model.getParticles();
                    const unsigned int nVert = triModels[i]->getParticleMesh().numVertices();
                    unsigned int offset = triModels[i]->getIndexOffset();
                    PBD::Simulation* sim = PBD::Simulation::getCurrent();
                    PBD::CubicSDFCollisionDetection* cd = dynamic_cast<PBD::CubicSDFCollisionDetection*>(sim->getTimeStep()->getCollisionDetection());
                    if (cd != nullptr)
                        cd->addCollisionObjectWithoutGeometry(i, PBD::CollisionDetection::CollisionObject::TriangleModelCollisionObjectType, &pd.getPosition(offset), nVert, true);
                }
                return triModels[i];
            }, py::arg("points"), py::arg("indices"), py::arg("uvIndices") = PBD::TriangleModel::ParticleMesh::UVIndices(),
                py::arg("uvs") = PBD::TriangleModel::ParticleMesh::UVs(), py::arg("testMesh") = false,
                py::return_value_policy::reference)
        .def("addRegularTriangleModel", [](PBD::SimulationModel &model, 
            const int width, const int height,
            const Vector3r& translation,
            const Matrix3r& rotation,
            const Vector2r& scale,
            const bool testMesh)
            {
                auto &triModels = model.getTriangleModels();
                int i = triModels.size();
                model.addRegularTriangleModel(width, height, translation, rotation, scale);
                if (testMesh)
                {
                    PBD::ParticleData& pd = model.getParticles();
                    const unsigned int nVert = triModels[i]->getParticleMesh().numVertices();
                    unsigned int offset = triModels[i]->getIndexOffset();
                    PBD::Simulation* sim = PBD::Simulation::getCurrent();
                    PBD::CubicSDFCollisionDetection* cd = dynamic_cast<PBD::CubicSDFCollisionDetection*>(sim->getTimeStep()->getCollisionDetection());
                    if (cd != nullptr)
                        cd->addCollisionObjectWithoutGeometry(i, PBD::CollisionDetection::CollisionObject::TriangleModelCollisionObjectType, &pd.getPosition(offset), nVert, true);
                }
                return triModels[i];
            }, py::arg("width"), py::arg("height"), py::arg("translation") = Vector3r::Zero(),
                py::arg("rotation") = Matrix3r::Identity(), py::arg("scale") = Vector2r::Ones(), py::arg("testMesh") = false,
                py::return_value_policy::reference)
        .def("addTetModel", [](
            PBD::SimulationModel& model,
            std::vector<Vector3r>& points,
            std::vector<unsigned int>& indices,
            const bool testMesh, 
            bool generateCollisionObject, const Eigen::Matrix<unsigned int, 3, 1>& resolution)
            {
                auto& tetModels = model.getTetModels();
                int i = tetModels.size();
                model.addTetModel(points.size(), indices.size()/4, points.data(), indices.data());

                PBD::ParticleData& pd = model.getParticles();
                PBD::TetModel* tetModel = tetModels[i];
                const unsigned int nVert = tetModel->getParticleMesh().numVertices();
                unsigned int offset = tetModel->getIndexOffset();
                PBD::Simulation* sim = PBD::Simulation::getCurrent();
                if (generateCollisionObject)
                {                   
                    auto &surfaceMesh = tetModel->getSurfaceMesh();
                    PBD::CubicSDFCollisionDetection::GridPtr sdf = generateSDF(points, surfaceMesh.getFaces(), resolution);
                    if (sdf != nullptr)
                    {
                        PBD::CubicSDFCollisionDetection* cd = dynamic_cast<PBD::CubicSDFCollisionDetection*>(sim->getTimeStep()->getCollisionDetection());
                        if (cd != nullptr)
                        {
                            auto index = cd->getCollisionObjects().size();
                            cd->addCubicSDFCollisionObject(i,
                                PBD::CollisionDetection::CollisionObject::TetModelCollisionObjectType,
                                &pd.getPosition(offset), nVert, sdf, Vector3r::Ones(), testMesh, false);

                            const unsigned int modelIndex = cd->getCollisionObjects()[index]->m_bodyIndex;
                            PBD::TetModel* tm = tetModels[modelIndex];
                            const unsigned int offset = tm->getIndexOffset();
                            const Utilities::IndexedTetMesh& mesh = tm->getParticleMesh();

                            ((PBD::DistanceFieldCollisionDetection::DistanceFieldCollisionObject*)cd->getCollisionObjects()[index])->initTetBVH(&pd.getPosition(offset), mesh.numVertices(), mesh.getTets().data(), mesh.numTets(), cd->getTolerance());
                        }
                    }
                }
                else if (testMesh)
                {                    
                    PBD::CubicSDFCollisionDetection* cd = dynamic_cast<PBD::CubicSDFCollisionDetection*>(sim->getTimeStep()->getCollisionDetection());
                    if (cd != nullptr)
                    {
                        auto index = cd->getCollisionObjects().size();
                        cd->addCollisionObjectWithoutGeometry(i, PBD::CollisionDetection::CollisionObject::TetModelCollisionObjectType, &pd.getPosition(offset), nVert, true);

                        const unsigned int modelIndex = cd->getCollisionObjects()[index]->m_bodyIndex;
                        PBD::TetModel* tm = tetModels[modelIndex];
                        const unsigned int offset = tm->getIndexOffset();
                        const Utilities::IndexedTetMesh& mesh = tm->getParticleMesh();

                        ((PBD::DistanceFieldCollisionDetection::DistanceFieldCollisionObject*)cd->getCollisionObjects()[index])->initTetBVH(&pd.getPosition(offset), mesh.numVertices(), mesh.getTets().data(), mesh.numTets(), cd->getTolerance());
                    }
                }
                return tetModel;
            }, py::arg("points"), py::arg("indices"), py::arg("testMesh") = false,
                py::arg("generateCollisionObject") = false, py::arg("resolution") = Eigen::Matrix<unsigned int, 3, 1>(30, 30, 30),
                py::return_value_policy::reference)
        .def("addRegularTetModel", [](PBD::SimulationModel &model, 
            const int width, const int height, const int depth,
            const Vector3r& translation,
            const Matrix3r& rotation,
            const Vector3r& scale,
            const bool testMesh)
            {
                auto &tetModels = model.getTetModels();
                int i = tetModels.size();
                model.addRegularTetModel(width, height, depth, translation, rotation, scale);
                if (testMesh)
                {
                    PBD::ParticleData& pd = model.getParticles();
                    const unsigned int nVert = tetModels[i]->getParticleMesh().numVertices();
                    unsigned int offset = tetModels[i]->getIndexOffset();
                    PBD::Simulation* sim = PBD::Simulation::getCurrent();
                    PBD::CubicSDFCollisionDetection* cd = dynamic_cast<PBD::CubicSDFCollisionDetection*>(sim->getTimeStep()->getCollisionDetection());
                    if (cd != nullptr)
                        cd->addCollisionObjectWithoutGeometry(i, PBD::CollisionDetection::CollisionObject::TetModelCollisionObjectType, &pd.getPosition(offset), nVert, true);
                }
                return tetModels[i];
            }, py::arg("width"), py::arg("height"), py::arg("depth"), py::arg("translation") = Vector3r::Zero(),
                py::arg("rotation") = Matrix3r::Identity(), py::arg("scale") = Vector3r::Ones(), py::arg("testMesh") = false, 
                py::return_value_policy::reference)
        .def("addLineModel", [](
            PBD::SimulationModel& model,
            const unsigned int nPoints,
            const unsigned int nQuaternions,
            std::vector<Vector3r>& points,
            std::vector<Quaternionr>& quaternions, 
            std::vector<unsigned int>& indices,
            std::vector<unsigned int>& indicesQuaternions)
            {
                model.addLineModel(nPoints, nQuaternions, points.data(), quaternions.data(), indices.data(), indicesQuaternions.data());
            })
        .def("addBallJoint", &PBD::SimulationModel::addBallJoint)
        .def("addBallOnLineJoint", &PBD::SimulationModel::addBallOnLineJoint)
        .def("addHingeJoint", &PBD::SimulationModel::addHingeJoint)
        .def("addTargetAngleMotorHingeJoint", &PBD::SimulationModel::addTargetAngleMotorHingeJoint)
        .def("addTargetVelocityMotorHingeJoint", &PBD::SimulationModel::addTargetVelocityMotorHingeJoint)
        .def("addUniversalJoint", &PBD::SimulationModel::addUniversalJoint)
        .def("addSliderJoint", &PBD::SimulationModel::addSliderJoint)
        .def("addTargetPositionMotorSliderJoint", &PBD::SimulationModel::addTargetPositionMotorSliderJoint)
        .def("addTargetVelocityMotorSliderJoint", &PBD::SimulationModel::addTargetVelocityMotorSliderJoint)
        .def("addRigidBodyParticleBallJoint", &PBD::SimulationModel::addRigidBodyParticleBallJoint)
        .def("addRigidBodySpring", &PBD::SimulationModel::addRigidBodySpring)
        .def("addDistanceJoint", &PBD::SimulationModel::addDistanceJoint)
        .def("addDamperJoint", &PBD::SimulationModel::addDamperJoint)
        .def("addRigidBodyContactConstraint", &PBD::SimulationModel::addRigidBodyContactConstraint)
        .def("addParticleRigidBodyContactConstraint", &PBD::SimulationModel::addParticleRigidBodyContactConstraint)
        .def("addParticleSolidContactConstraint", &PBD::SimulationModel::addParticleSolidContactConstraint)
        .def("addDistanceConstraint", &PBD::SimulationModel::addDistanceConstraint)
        .def("addDistanceConstraint_XPBD", &PBD::SimulationModel::addDistanceConstraint_XPBD)
        .def("addDihedralConstraint", &PBD::SimulationModel::addDihedralConstraint)
        .def("addIsometricBendingConstraint", &PBD::SimulationModel::addIsometricBendingConstraint)
        .def("addIsometricBendingConstraint_XPBD", &PBD::SimulationModel::addIsometricBendingConstraint_XPBD)
        .def("addFEMTriangleConstraint", &PBD::SimulationModel::addFEMTriangleConstraint)
        .def("addStrainTriangleConstraint", &PBD::SimulationModel::addStrainTriangleConstraint)
        .def("addVolumeConstraint", &PBD::SimulationModel::addVolumeConstraint)
        .def("addVolumeConstraint_XPBD", &PBD::SimulationModel::addVolumeConstraint_XPBD)
        .def("addFEMTetConstraint", &PBD::SimulationModel::addFEMTetConstraint)
        .def("addStrainTetConstraint", &PBD::SimulationModel::addStrainTetConstraint)
        .def("addShapeMatchingConstraint", [](
            PBD::SimulationModel& model,
            const unsigned int numberOfParticles, 
            const std::vector<unsigned int>& particleIndices,
            const std::vector<unsigned int>& numClusters,
            const Real stiffness)
            {
                model.addShapeMatchingConstraint(numberOfParticles, particleIndices.data(), numClusters.data(), stiffness);
            })


        .def("addStretchShearConstraint", &PBD::SimulationModel::addStretchShearConstraint)
        .def("addBendTwistConstraint", &PBD::SimulationModel::addBendTwistConstraint)
        .def("addStretchBendingTwistingConstraint", &PBD::SimulationModel::addStretchBendingTwistingConstraint)
        .def("addDirectPositionBasedSolverForStiffRodsConstraint", &PBD::SimulationModel::addDirectPositionBasedSolverForStiffRodsConstraint)
        
        .def("getParticles", &PBD::SimulationModel::getParticles, py::return_value_policy::reference)
        .def("getRigidBodies", &PBD::SimulationModel::getRigidBodies, py::return_value_policy::reference)
        .def("getTriangleModels", &PBD::SimulationModel::getTriangleModels, py::return_value_policy::reference)
        .def("getTetModels", &PBD::SimulationModel::getTetModels, py::return_value_policy::reference)
        .def("getLineModels", &PBD::SimulationModel::getLineModels, py::return_value_policy::reference)
        .def("getConstraints", &PBD::SimulationModel::getConstraints, py::return_value_policy::reference)
        .def("getOrientations", &PBD::SimulationModel::getOrientations, py::return_value_policy::reference)
        .def("getRigidBodyContactConstraints", &PBD::SimulationModel::getRigidBodyContactConstraints, py::return_value_policy::reference)
        .def("getParticleRigidBodyContactConstraints", &PBD::SimulationModel::getParticleRigidBodyContactConstraints, py::return_value_policy::reference)
        .def("getParticleSolidContactConstraints", &PBD::SimulationModel::getParticleSolidContactConstraints, py::return_value_policy::reference)
        .def("getConstraintGroups", &PBD::SimulationModel::getConstraintGroups, py::return_value_policy::reference)
        .def("resetContacts", &PBD::SimulationModel::resetContacts)

        .def("addClothConstraints", &PBD::SimulationModel::addClothConstraints)
        .def("addBendingConstraints", &PBD::SimulationModel::addBendingConstraints)
        .def("addSolidConstraints", &PBD::SimulationModel::addSolidConstraints)

        .def("getContactStiffnessRigidBody", &PBD::SimulationModel::getContactStiffnessRigidBody)
        .def("setContactStiffnessRigidBody", &PBD::SimulationModel::setContactStiffnessRigidBody)
        .def("getContactStiffnessParticleRigidBody", &PBD::SimulationModel::getContactStiffnessParticleRigidBody)
        .def("setContactStiffnessParticleRigidBody", &PBD::SimulationModel::setContactStiffnessParticleRigidBody)

        .def("addRigidBody", [](PBD::SimulationModel &model, const Real density, 
            const PBD::VertexData& vertices, 
            const Utilities::IndexedFaceMesh& mesh, 
            const Vector3r& translation, const Matrix3r& rotation,
            const Vector3r& scale, 
            const bool testMesh,
            const PBD::CubicSDFCollisionDetection::GridPtr sdf)
            {
                PBD::SimulationModel::RigidBodyVector& rbs = model.getRigidBodies();
                PBD::RigidBody *rb = new PBD::RigidBody();
                rb->initBody(density, translation, Quaternionr(rotation), vertices, mesh, scale);
                rbs.push_back(rb);
                if (sdf != nullptr)
                {
                    PBD::Simulation* sim = PBD::Simulation::getCurrent();
                    PBD::CubicSDFCollisionDetection* cd = dynamic_cast<PBD::CubicSDFCollisionDetection*>(sim->getTimeStep()->getCollisionDetection());
                    if (cd != nullptr)
                    {
                        const std::vector<Vector3r>& vertices = rb->getGeometry().getVertexDataLocal().getVertices();
                        const unsigned int nVert = static_cast<unsigned int>(vertices.size());
                        cd->addCubicSDFCollisionObject(rbs.size() - 1,
                            PBD::CollisionDetection::CollisionObject::RigidBodyCollisionObjectType,
                            vertices.data(), nVert, sdf, scale, testMesh, false);
                    }
                }
                return rb;
            }, py::arg("density"), py::arg("vertices"), py::arg("mesh"), py::arg("translation") = Vector3r::Zero(),
                py::arg("rotation") = Matrix3r::Identity(), py::arg("scale") = Vector3r::Ones(), py::arg("testMesh") = false,
                py::arg("sdf"),
                py::return_value_policy::reference)
        .def("addRigidBody", [](PBD::SimulationModel &model, const Real density, 
            const PBD::VertexData& vertices, 
            const Utilities::IndexedFaceMesh& mesh, 
            const Vector3r& translation, const Matrix3r &rotation,
            const Vector3r& scale,             
            const bool testMesh, 
            const bool generateCollisionObject, const Eigen::Matrix<unsigned int, 3, 1>& resolution)
            {
                PBD::Simulation* sim = PBD::Simulation::getCurrent();
                PBD::SimulationModel::RigidBodyVector& rbs = model.getRigidBodies();
                PBD::RigidBody *rb = new PBD::RigidBody();
                auto i = rbs.size();
                rb->initBody(density, translation, Quaternionr(rotation), vertices, mesh, scale);
                rbs.push_back(rb);

                if (generateCollisionObject)
                {
                    PBD::CubicSDFCollisionDetection::GridPtr sdf = generateSDF(vertices.getVertices(), mesh.getFaces(), resolution);
                    if (sdf != nullptr)
                    {
                        PBD::CubicSDFCollisionDetection* cd = dynamic_cast<PBD::CubicSDFCollisionDetection*>(sim->getTimeStep()->getCollisionDetection());
                        if (cd != nullptr)
                        {
                            const std::vector<Vector3r>& vertices = rb->getGeometry().getVertexDataLocal().getVertices();
                            const unsigned int nVert = static_cast<unsigned int>(vertices.size());
                            cd->addCubicSDFCollisionObject(rbs.size() - 1,
                                PBD::CollisionDetection::CollisionObject::RigidBodyCollisionObjectType,
                                vertices.data(), nVert, sdf, scale, testMesh, false);
                        }
                    }
                }
                else if (testMesh)
                {
                    PBD::CubicSDFCollisionDetection* cd = dynamic_cast<PBD::CubicSDFCollisionDetection*>(sim->getTimeStep()->getCollisionDetection());
                    if (cd != nullptr)
                    {
                        auto index = cd->getCollisionObjects().size();
                        const std::vector<Vector3r>& vertices = rbs[i]->getGeometry().getVertexDataLocal().getVertices();
                        const unsigned int nVert = static_cast<unsigned int>(vertices.size());
                        cd->addCollisionObjectWithoutGeometry(i, PBD::CollisionDetection::CollisionObject::RigidBodyCollisionObjectType, vertices.data(), nVert, true);
                    }
                }
                return rb;
            }, py::arg("density"), py::arg("vertices"), py::arg("mesh"), py::arg("translation") = Vector3r::Zero(),
                py::arg("rotation") = Matrix3r::Identity(), py::arg("scale") = Vector3r::Ones(), py::arg("testMesh") = false,
                py::arg("generateCollisionObject") = false, py::arg("resolution") = Eigen::Matrix<unsigned int, 3, 1>(30,30,30),
                py::return_value_policy::reference)
        .def("setActor", [](PBD::SimulationModel &model, const std::string& model_path, std::array<float, 10> shape, const Eigen::Matrix<unsigned int, 3, 1>& resolution, const Real scale, std::pair<float, float>& extension)
            {
				// load actor
				smplx::Gender gender = smplx::Gender::neutral;
				smpl_model = std::make_unique<smplx::ModelX>(model_path, "", gender);
				smpl_body = std::make_unique<smplx::BodyX>(*smpl_model);
				smpl_body->shape().setZero();
				smpl_body->pose().setZero();
				smpl_body->update(false, true);
				resolution_ = resolution;
				scale_ = scale;
				extension_ = extension;

				// init vd, mesh for full smpl mesh for visual representation
				smpl_to_mesh(vd_smpl_full, mesh_smpl_full, scale);

				// init vd, mesh for culled smpl mesh for collision
				smpl_to_mesh_torso(vd_smpl_culled, mesh_smpl_culled, scale);

				// apply betas after the cull
				smpl_body->shape().head<10>() = Eigen::Map<Eigen::VectorXf>(shape.data(), shape.size());
				smpl_body->update(false, true);
				mesh_vertex_update(vd_smpl_full, scale_);
				mesh_vertex_update_cull(vd_smpl_culled, scale_);

				// check if the actor already exists
				PBD::RigidBody* rb_actor = nullptr;
				for (auto& rb : model.getRigidBodies()) {
					if (rb->getName() == "rb_actor") {
						rb_actor = rb;
						break;
					}
				}

				// new one if not
				if (rb_actor == nullptr) {
					rb_actor = new PBD::RigidBody();
					// init rb with mesh data
					rb_actor->initBody(1.0, Vector3r(0.0, 0.0, 0.0), Quaternionr(1.0, 0.0, 0.0, 0.0), vd_smpl_culled,
									mesh_smpl_culled, Vector3r(scale, scale, scale));
					rb_actor->setName("rb_actor");
					rb_actor->setMass(0.0);
					rb_actor->setFrictionCoeff(static_cast<Real>(0.9));
					model.getRigidBodies().push_back(rb_actor);
				}

				// check if the actor sdf already exists
				auto* scd = dynamic_cast<PBD::CubicSDFCollisionDetection*>(
					PBD::Simulation::getCurrent()->getTimeStep()->getCollisionDetection());
				PBD::CollisionDetection::CollisionObject* co_actor = nullptr;
				for (auto& co : scd->getCollisionObjects()) {
					if (co->getName() == "co_actor") {
						co_actor = co;
						break;
					}
				}

				// for now, just delete and create a new collision object
				if (co_actor != nullptr) {
					auto& collisionObjects = scd->getCollisionObjects();
					collisionObjects.erase(
						std::remove(collisionObjects.begin(), collisionObjects.end(), co_actor),
						collisionObjects.end());
					delete co_actor;
					co_actor = nullptr;
				}

				auto sdf = generateSDF(vd_smpl_culled.getVertices(), mesh_smpl_culled.getFaces(), resolution_, extension_);
				const auto vertices = vd_smpl_culled.getVertices();
				const unsigned int nVert = static_cast<unsigned int>(vertices.size());
				scd->addCubicSDFCollisionObject(
					model.getRigidBodies().size() - 1,
					PBD::CollisionDetection::CollisionObject::RigidBodyCollisionObjectType, vertices.data(),
					nVert, sdf, Vector3r(scale_, scale_, scale_), true, false, "co_actor");

				return rb_actor;
            }, py::arg("model_path"), py::arg("shape"), py::arg("resolution") = Eigen::Matrix<unsigned int, 3, 1>(20, 20, 20), py::arg("scale") = 1.0, py::arg("extension") = std::make_pair(0.0f, 0.0f))
        .def("setActorPose", [](PBD::SimulationModel &model, std::array<float, 66> pose)
            { 
				smpl_body->pose().setZero();
				smpl_body->pose().head<66>() = Eigen::Map<Eigen::VectorXf>(pose.data(), pose.size());
				smpl_body->pose().head<3>().setZero();
				smpl_body->trans() = Eigen::Vector3f(0, 0, 0);
				// _SMPLX_BEGIN_PROFILE;
				smpl_body->update(false, true);
				// _SMPLX_PROFILE(update time);

				// called every frame
				mesh_vertex_update_cull(vd_smpl_culled, scale_);

				// check if the actor already exists
				PBD::RigidBody* rb_actor = nullptr;
				for (auto& rb : model.getRigidBodies()) {
					if (rb->getName() == "rb_actor") {
						rb_actor = rb;
						break;
					}
				}

				// new one if there isn't one
				if (rb_actor == nullptr) {
					rb_actor = new PBD::RigidBody();
					// init rb with mesh data
					rb_actor->initBody(1.0, Vector3r(0.0, 0.0, 0.0), Quaternionr(1.0, 0.0, 0.0, 0.0), vd_smpl_culled,
									mesh_smpl_culled, Vector3r(scale_, scale_, scale_));
					rb_actor->setName("rb_actor");
					rb_actor->setMass(0.0);
					rb_actor->setFrictionCoeff(static_cast<Real>(0.9));
					model.getRigidBodies().push_back(rb_actor);
				}

				// apply skinned smpl model to rb
				if (rb_actor != nullptr) {
					mesh_vertex_update(vd_smpl_full, scale_ * scale_);
					mesh_vertex_update_cull(rb_actor->getGeometry().getVertexData(), scale_ * scale_);
				}

				// check if the actor sdf already exists
				auto* scd = dynamic_cast<PBD::CubicSDFCollisionDetection*>(
					PBD::Simulation::getCurrent()->getTimeStep()->getCollisionDetection());
				PBD::CollisionDetection::CollisionObject* co_actor = nullptr;
				for (auto& co : scd->getCollisionObjects()) {
					if (co->getName() == "co_actor") {
						co_actor = co;
						break;
					}
				}

				// for now, just delete and create a new collision object
				if (co_actor != nullptr) {
					auto& collisionObjects = scd->getCollisionObjects();
					collisionObjects.erase(
						std::remove(collisionObjects.begin(), collisionObjects.end(), co_actor),
						collisionObjects.end());
					delete co_actor;
					co_actor = nullptr;
				}

				auto sdf = generateSDF(vd_smpl_culled.getVertices(), mesh_smpl_culled.getFaces(), resolution_, extension_);
				const auto vertices = vd_smpl_culled.getVertices();
				const unsigned int nVert = static_cast<unsigned int>(vertices.size());
				scd->addCubicSDFCollisionObject(
					model.getRigidBodies().size() - 1,
					PBD::CollisionDetection::CollisionObject::RigidBodyCollisionObjectType, vertices.data(),
					nVert, sdf, Vector3r(scale_, scale_, scale_), true, false, "co_actor");
            }, py::arg("pose"))
        .def("getActorGeometry", [](PBD::SimulationModel &model)
            {
                if (vd_smpl_full.getVertices().empty() || mesh_smpl_full.getFaces().empty()) {
                    throw std::runtime_error("VertexData or IndexedFaceMesh is not initialized properly");
                }
                return std::make_pair(std::ref(vd_smpl_full), std::ref(mesh_smpl_full));
            })
        .def("createMesh", [](PBD::SimulationModel &model, const py::array_t<Real>& verticesArray, const py::array_t<unsigned int>& facesArray, const Real& scale)
            {
                PBD::VertexData vd;
                Utilities::IndexedFaceMesh mesh;

                py::buffer_info verticesInfo = verticesArray.request();
                py::buffer_info facesInfo = facesArray.request();

                Real* verticesPtr = static_cast<Real*>(verticesInfo.ptr);
                unsigned int* facesPtr = static_cast<unsigned int*>(facesInfo.ptr);

                const unsigned int nPoints = static_cast<unsigned int>(verticesInfo.size / 3);
                const unsigned int nFaces = static_cast<unsigned int>(facesInfo.size / 3);

                mesh.initMesh(nPoints, nFaces * 2, nFaces);
                vd.reserve(nPoints);

                for (size_t i = 0; i < verticesInfo.size; i += 3) {
                    vd.addVertex(Vector3r(verticesPtr[i], verticesPtr[i + 1], verticesPtr[i + 2]) * scale);
                }

                for (size_t i = 0; i < facesInfo.size; i += 3) {
                    unsigned int indices[3] = {facesPtr[i], facesPtr[i + 1], facesPtr[i + 2]};
                    mesh.addFace(indices);
                }

                mesh.buildNeighbors();
                mesh.updateNormals(vd, 0);
                mesh.updateVertexNormals(vd);

                return std::make_pair(vd, mesh);
            }, py::arg("vertices"), py::arg("faces"), py::arg("scale") = 1.0)
        .def("querySDF", [](PBD::SimulationModel &model, PBD::RigidBody *rb, const Vector3r& x_w)
            {
				// check if the actor sdf already exists
				auto* scd = dynamic_cast<PBD::CubicSDFCollisionDetection*>(
					PBD::Simulation::getCurrent()->getTimeStep()->getCollisionDetection());
				PBD::CollisionDetection::CollisionObject* co_actor = nullptr;
				for (auto& co : scd->getCollisionObjects()) {
					if (co->getName() == "co_actor") {
						co_actor = co;
						break;
					}
				}

				PBD::CubicSDFCollisionDetection::CubicSDFCollisionObject* dfco =
					dynamic_cast<PBD::CubicSDFCollisionDetection::CubicSDFCollisionObject*>(co_actor);
				if (!dfco) {
					throw std::runtime_error(
						"The collision object is not a CubicSDFCollisionDetection::CubicSDFCollisionObject.");
				}

				const Vector3r& com = rb->getPosition();
				const Matrix3r& R = rb->getTransformationR();
				const Vector3r& v1 = rb->getTransformationV1();
				const Vector3r& v2 = rb->getTransformationV2();

				// x world -> x local -> query -> (distance, n_local) -> (distance, n_world)
				const Vector3r x = R * (x_w - com) + v1;
				Vector3r cp, n;
				Real dist;
				Real max_dist = 10.0;
				dfco->collisionTest(x, 0.0, cp, n, dist, max_dist);

				const Vector3r n_w = R.transpose() * n;

				return std::make_tuple(dist, n_w);
            }, py::arg("rigid_body"), py::arg("query_point"))
        .def("querySDFBatch", [](PBD::SimulationModel &model, PBD::RigidBody *rb, const std::vector<Vector3r>& x_w_batch)
            {
                py::gil_scoped_release release;

				std::vector<Real> dist_batch;
				std::vector<Vector3r> n_w_batch;
				std::vector<Vector3r> cp_batch;

				// check if the actor sdf already exists
				auto* scd = dynamic_cast<PBD::CubicSDFCollisionDetection*>(
					PBD::Simulation::getCurrent()->getTimeStep()->getCollisionDetection());
				PBD::CollisionDetection::CollisionObject* co_actor = nullptr;
				for (auto& co : scd->getCollisionObjects()) {
					if (co->getName() == "co_actor") {
						co_actor = co;
						break;
					}
				}

				PBD::CubicSDFCollisionDetection::CubicSDFCollisionObject* dfco =
					dynamic_cast<PBD::CubicSDFCollisionDetection::CubicSDFCollisionObject*>(co_actor);
				if (!dfco) {
					throw std::runtime_error(
						"The collision object is not a CubicSDFCollisionDetection::CubicSDFCollisionObject.");
				}

				const Vector3r& com = rb->getPosition();
				const Matrix3r& R = rb->getTransformationR();
				const Vector3r& v1 = rb->getTransformationV1();
				const Vector3r& v2 = rb->getTransformationV2();

				std::vector<Vector3r> x_batch;
				x_batch.reserve(x_w_batch.size());
				for (const Vector3r& x_w : x_w_batch) {
					// x world -> x local -> query -> (distance, n_local) -> (distance, n_world)
					const Vector3r x = R * (x_w - com) + v1;
					x_batch.push_back(x);
				}

				Real max_dist = 10.0;
				bool collision =
					dfco->collisionTestBatch(x_batch, 0.0, cp_batch, n_w_batch, dist_batch, max_dist);

				for (size_t i = 0; i < cp_batch.size(); ++i) {
					if (n_w_batch[i] != Vector3r(0, 0, 0)) {
						// local to world conversion
						n_w_batch[i] = R.transpose() * n_w_batch[i];
					} 
				}

				return std::make_tuple(dist_batch, n_w_batch);
            }, py::arg("rigid_body"), py::arg("query_points"))
        ;

}