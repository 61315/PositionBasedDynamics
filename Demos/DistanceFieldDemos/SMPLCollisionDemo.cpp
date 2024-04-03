#include <Eigen/Dense>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>

#include "Common/Common.h"
#include "Demos/Common/DemoBase.h"
#include "Demos/Visualization/MiniGL.h"
#include "Demos/Visualization/Selection.h"
#include "Demos/Visualization/Visualization.h"
#include <Simulation/CubicSDFCollisionDetection.h>
#include "Simulation/DistanceFieldCollisionDetection.h"
#include "Simulation/Simulation.h"
#include "Simulation/SimulationModel.h"
#include "Simulation/TimeManager.h"
#include "Simulation/TimeStepController.h"
#include "Utils/FileSystem.h"
#include "Utils/Logger.h"
#include "Utils/Timing.h"
#include "smplx/smplx.hpp"
#include "smplx/util.hpp"

#if defined(_DEBUG) && !defined(EIGEN_ALIGN)
#define new DEBUG_NEW
#endif

using namespace PBD;
using namespace Eigen;
using namespace std;
using namespace Utilities;
using namespace smplx;

void timeStep();
void buildModel();
void createTriangleModels();
void render();
void reset();

const int nRows = 50;
const int nCols = 50;
const Real width = 10.0;
const Real height = 10.0;
bool doPause = true;
DemoBase* base;
DistanceFieldCollisionDetection* cd;
bool test1 = false;

IndexedFaceMesh mesh_smpl_full;
VertexData vd_smpl_full;

IndexedFaceMesh mesh_smpl_culled;
VertexData vd_smpl_culled;

// locate the "SMPLX_NEUTRAL.npz" file here
const std::string SMPL_MODEL_PATH = "data/models/smplx/SMPLX_NEUTRAL.npz";

std::unique_ptr<smplx::BodyX> smpl_body;
std::unique_ptr<smplx::BodyX> smpl_body_dummy;
std::unique_ptr<smplx::ModelX> smpl_model;

std::vector<std::array<int, 3>> filtered_faces;
std::vector<int> filtered_face_indices;
std::vector<int> filtered_vertex_indices;

Eigen::Matrix<unsigned int, 3, 1> resolution_(20, 20, 20);
Real scale_ = 2;
std::pair<float, float>& extension_ = std::make_pair(0.0f, 0.0f);

Vector3f start_trans(0, 0, 0);

PBD::CubicSDFCollisionDetection::GridPtr generateSDF(
 	const std::vector<Vector3r>& vertices, const std::vector<unsigned int>& faces,
 	const Eigen::Matrix<unsigned int, 3, 1>& resolution,
 	const std::pair<float, float>& extension = std::make_pair(0.0f, 0.0f)) {
 	const unsigned int nFaces = faces.size() / 3;
 #ifdef USE_DOUBLE
 	Discregrid::TriangleMesh sdfMesh((vertices[0]).data(), faces.data(), vertices.size(), nFaces);
 #else
 	// if type is float, copy vector to double vector
 	std::vector<double> doubleVec;
 	doubleVec.resize(3 * vertices.size());
 	for (unsigned int i = 0; i < vertices.size(); i++)
 		for (unsigned int j = 0; j < 3; j++) doubleVec[3 * i + j] = vertices[i][j];
 	Discregrid::TriangleMesh sdfMesh(&doubleVec[0], faces.data(), vertices.size(), nFaces);
 #endif
 	Discregrid::TriangleMeshDistance md(sdfMesh);
 	Eigen::AlignedBox3d domain;
 	for (auto const& x : sdfMesh.vertices()) {
 		domain.extend(x);
 	}
 	domain.max() += 0.1 * Eigen::Vector3d::Ones();
 	domain.min() -= 0.1 * Eigen::Vector3d::Ones();

 	domain.min() -= extension.first * Eigen::Vector3d::Ones();
 	domain.max() += extension.second * Eigen::Vector3d::Ones();

 	std::cout << "Set SDF resolution: " << resolution[0] << ", " << resolution[1] << ", "
 			  << resolution[2] << std::endl;
 	auto sdf = std::make_shared<PBD::CubicSDFCollisionDetection::Grid>(
 		domain, std::array<unsigned int, 3>({resolution[0], resolution[1], resolution[2]}));
 	auto func = Discregrid::DiscreteGrid::ContinuousFunction{};
 	func = [&md](Eigen::Vector3d const& xi) { return md.signed_distance(xi).distance; };
 	std::cout << "Generate SDF\n";
 	sdf->addFunction(func, true);
	return sdf;
}

PBD::TriangleModel* addTriangleModel(PBD::SimulationModel& model, std::vector<Vector3r>& points,
									 std::vector<unsigned int>& indices,
									 const bool testMesh = false) {
	const PBD::TriangleModel::ParticleMesh::UVIndices& uvIndices =
		PBD::TriangleModel::ParticleMesh::UVIndices();
	const PBD::TriangleModel::ParticleMesh::UVs& uvs = PBD::TriangleModel::ParticleMesh::UVs();
	auto& triModels = model.getTriangleModels();
	int i = triModels.size();
	model.addTriangleModel(points.size(), indices.size() / 3, points.data(), indices.data(),
						   uvIndices, uvs);
	if (testMesh) {
		PBD::ParticleData& pd = model.getParticles();
		const unsigned int nVert = triModels[i]->getParticleMesh().numVertices();
		unsigned int offset = triModels[i]->getIndexOffset();
		PBD::Simulation* sim = PBD::Simulation::getCurrent();
		PBD::CubicSDFCollisionDetection* cd = dynamic_cast<PBD::CubicSDFCollisionDetection*>(
			sim->getTimeStep()->getCollisionDetection());
		if (cd != nullptr)
			cd->addCollisionObjectWithoutGeometry(
				i, PBD::CollisionDetection::CollisionObject::TriangleModelCollisionObjectType,
				&pd.getPosition(offset), nVert, true);
	}
	return triModels[i];
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

std::pair<VertexData&, IndexedFaceMesh&> getActorGeometry() {
	if (vd_smpl_full.getVertices().empty() || mesh_smpl_full.getFaces().empty()) {
		throw std::runtime_error("VertexData or IndexedFaceMesh is not initialized properly");
	}
	return std::make_pair(std::ref(vd_smpl_full), std::ref(mesh_smpl_full));
}

PBD::RigidBody* setActor(PBD::SimulationModel& model, const std::string& model_path,
						 std::array<float, 10> shape,
						 const Eigen::Matrix<unsigned int, 3, 1>& resolution, const Real scale,
						 std::pair<float, float>& extension = std::make_pair(0.0f, 0.0f)) {
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
		rb_actor->initBody(1.0, Vector3r(0.0, 0.0, 0.0), Quaternionr(1.0, 0.0, 0.0, 0.0),
						   vd_smpl_culled, mesh_smpl_culled, Vector3r(scale, scale, scale));
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

	auto sdf = generateSDF(vd_smpl_culled.getVertices(), mesh_smpl_culled.getFaces(), resolution_,
						   extension_);
	const auto vertices = vd_smpl_culled.getVertices();
	const unsigned int nVert = static_cast<unsigned int>(vertices.size());
	scd->addCubicSDFCollisionObject(
		model.getRigidBodies().size() - 1,
		PBD::CollisionDetection::CollisionObject::RigidBodyCollisionObjectType, vertices.data(),
		nVert, sdf, Vector3r(scale_, scale_, scale_), true, false, "co_actor");

	return rb_actor;
}

void setActorPose(PBD::SimulationModel& model, std::array<float, 66> pose) {
	smpl_body->pose().setZero();
	smpl_body->pose().head<66>() = Eigen::Map<Eigen::VectorXf>(pose.data(), pose.size());
	smpl_body->pose().head<3>().setZero();
	smpl_body->trans() = Vector3f(0, 0, 0);
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
		rb_actor->initBody(1.0, Vector3r(0.0, 0.0, 0.0), Quaternionr(1.0, 0.0, 0.0, 0.0),
						   vd_smpl_culled, mesh_smpl_culled, Vector3r(scale_, scale_, scale_));
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

	auto sdf = generateSDF(vd_smpl_culled.getVertices(), mesh_smpl_culled.getFaces(), resolution_,
						   extension_);
	const auto vertices = vd_smpl_culled.getVertices();
	const unsigned int nVert = static_cast<unsigned int>(vertices.size());
	scd->addCubicSDFCollisionObject(
		model.getRigidBodies().size() - 1,
		PBD::CollisionDetection::CollisionObject::RigidBodyCollisionObjectType, vertices.data(),
		nVert, sdf, Vector3r(scale_, scale_, scale_), true, false, "co_actor");
}

std::tuple<Real, Vector3r> querySDF(PBD::SimulationModel& model, PBD::RigidBody* rb,
									const Vector3r& x_w) {
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
}

std::tuple<std::vector<Real>, std::vector<Vector3r>> querySDFBatch(
	PBD::SimulationModel& model, PBD::RigidBody* rb, const std::vector<Vector3r>& x_w_batch) {
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
}

void buildActor() {
	SimulationModel* model = Simulation::getCurrent()->getModel();
	const std::string model_path = SMPL_MODEL_PATH;  
	std::array<float, 10> shape = {2, 0, -.91, 0.082, 1, -2, .832, -0.106, 0.644, 0.548};
	setActor(*model, model_path, shape, Eigen::Matrix<unsigned int, 3, 1>(20, 20, 20), scale_,
			 std::make_pair(0.0f, 0.0f));
	// setDummyActor();

	std::array<float, 66> rest_pose;
	rest_pose.fill(0.0f);
	rest_pose[3 * 16 + 2] = -0.333f;
	rest_pose[3 * 17 + 2] = 0.333f;

	setActorPose(*model, rest_pose);
}

void loadMesh(const std::string& file) {
	SimulationModel* model = Simulation::getCurrent()->getModel();
	IndexedFaceMesh mesh;
	VertexData vd;
	DemoBase::loadMesh(file, vd, mesh, Vector3r(0.0, -0.3, 0.0), Matrix3r::Identity(),
					   Vector3r(4, 4, 4));
	model->addTriangleModel(vd.size(), mesh.numFaces(), &vd.getPosition(0), mesh.getFaces().data(),
							mesh.getUVIndices(), mesh.getUVs());
}

/** Create a particle model mesh
 */
void createTriangleModels() {
	SimulationModel* model = Simulation::getCurrent()->getModel();

	string shirts_file = FileSystem::normalizePath(base->getExePath() +
												   "/resources/models/shirts.obj");
	loadMesh(shirts_file);

	string vest_file =
		FileSystem::normalizePath(base->getExePath() + "/resources/models/vest.obj");
	loadMesh(vest_file);

	// init constraints
	for (unsigned int cm = 0; cm < model->getTriangleModels().size(); cm++) {
		model->setClothStiffness(1.0);
		if (model->getClothSimulationMethod() == 4) model->setClothStiffness(100000);

		model->addClothConstraints(
			model->getTriangleModels()[cm], model->getClothSimulationMethod(),
			model->getClothStiffness(), model->getClothStiffnessXX(), model->getClothStiffnessYY(),
			model->getClothStiffnessXY(), model->getClothPoissonRatioXY(),
			model->getClothPoissonRatioYX(), model->getClothNormalizeStretch(),
			model->getClothNormalizeShear());

		model->setClothBendingStiffness(0.01);
		if (model->getClothBendingMethod() == 3) model->setClothBendingStiffness(10.0);

		model->addBendingConstraints(model->getTriangleModels()[cm], model->getClothBendingMethod(),
									 model->getClothBendingStiffness());
	}

	LOG_INFO << "?Number of triangles: "
			 << model->getTriangleModels()[0]->getParticleMesh().numFaces();
	LOG_INFO << "Number of vertices: " << nRows * nCols;
}

// main
int main(int argc, char** argv) {
	REPORT_MEMORY_LEAKS

	base = new DemoBase();
	base->init(argc, argv, "Cloth collision demo");

	SimulationModel* model = new SimulationModel();
	model->init();
	Simulation::getCurrent()->setModel(model);

	base->createParameterGUI();
	// base->setValue(DemoBase::RENDER_SDF, true);

	// reset simulation when cloth simulation/bending method has changed
	model->setValue(SimulationModel::CLOTH_SIMULATION_METHOD,
					SimulationModel::ENUM_CLOTHSIM_DISTANCE_CONSTRAINTS_XPBD);
	model->setClothBendingMethod(SimulationModel::ENUM_CLOTH_BENDING_ISOMETRIX_XPBD);
	model->setContactStiffnessParticleRigidBody(100);
	model->setClothStiffness(100000);

	model->setClothSimulationMethodChangedCallback([&]() { reset(); });
	model->setClothBendingMethodChangedCallback([&]() { reset(); });

	buildModel();

	// OpenGL
	MiniGL::setClientIdleFunc(timeStep);
	MiniGL::addKeyFunc('r', reset);
	MiniGL::setClientSceneFunc(render);
	MiniGL::setViewport(40.0f, 0.1f, 500.0f, Vector3r(0.0, 5, 10.0),
						Vector3r(0.0, 0.0, 0.0));  // undo x ccw 90
	// MiniGL::setViewport (40.0f, 0.1f, 500.0f, Vector3r (0.0, -10, 1.0), Vector3r (0.0, 0.0,
	// 0.0)); // undo x ccw 90

	MiniGL::mainLoop();

	base->cleanup();

	Utilities::Timing::printAverageTimes();
	Utilities::Timing::printTimeSums();

	delete Simulation::getCurrent();
	delete base;
	delete model;

	return 0;
}

void reset() {
	Utilities::Timing::printAverageTimes();
	Utilities::Timing::reset();

	Simulation::getCurrent()->reset();
	base->getSelectedParticles().clear();

	Simulation::getCurrent()->getModel()->cleanup();
	cd->cleanup();

	buildModel();
}

void timeStep() {
	const Real pauseAt = base->getValue<Real>(DemoBase::PAUSE_AT);
	if ((pauseAt > 0.0) && (pauseAt < TimeManager::getCurrent()->getTime()))
		base->setValue(DemoBase::PAUSE, true);

	if (base->getValue<bool>(DemoBase::PAUSE)) return;

	// Simulation code
	SimulationModel* model = Simulation::getCurrent()->getModel();
	auto* scd = dynamic_cast<PBD::CubicSDFCollisionDetection*>(
		Simulation::getCurrent()->getTimeStep()->getCollisionDetection());

	RigidBody* rb_actor = nullptr;
	for (auto& rb : model->getRigidBodies()) {
		if (rb->getName() == "rb_actor") {
			rb_actor = rb;
			break;
		}
	}

	if (rb_actor != nullptr && test1 == false) {
		static int frameCounter = 0;
		static bool forward = true;
		auto t0_construction = std::chrono::high_resolution_clock::now();
		if (TimeManager::getCurrent()->getTime() > 3.0) {
			auto sim = PBD::Simulation::getCurrent();

			double current_time = PBD::TimeManager::getCurrent()->getTime() - 3;
			double wave = std::sin(current_time) / 8;

			std::array<float, 66> pose;
			pose.fill(static_cast<float>(wave));
			pose[3 * 16 + 2] = static_cast<float>(-0.333 + wave);
			pose[3 * 17 + 2] = static_cast<float>(0.333 + wave);

			setActorPose(*model, pose);
		}
		std::cout << "\rDynamic SDF took: " << std::setw(15)
				  << static_cast<double>(
						 std::chrono::duration_cast<std::chrono::milliseconds>(
							 std::chrono::high_resolution_clock::now() - t0_construction)
							 .count()) /
						 1000.0
				  << "s" << std::endl;
	}

	const unsigned int numSteps = base->getValue<unsigned int>(DemoBase::NUM_STEPS_PER_RENDER);
	for (unsigned int i = 0; i < numSteps; i++) {
		START_TIMING("SimStep");
		Simulation::getCurrent()->getTimeStep()->step(*model);
		STOP_TIMING_AVG;

		base->step();
	}

	for (unsigned int i = 0; i < model->getTriangleModels().size(); i++)
		model->getTriangleModels()[i]->updateMeshNormals(model->getParticles());
}

void buildModel() {
	TimeManager::getCurrent()->setTimeStepSize(static_cast<Real>(0.005));

	SimulationModel* model = Simulation::getCurrent()->getModel();
	SimulationModel::RigidBodyVector& rb = model->getRigidBodies();

	cd = new PBD::CubicSDFCollisionDetection();
	Simulation::getCurrent()->getTimeStep()->setCollisionDetection(*model, cd);
	cd->setTolerance(static_cast<Real>(0.025));

	buildActor();

	createTriangleModels();

	// collision object for cloth mesh(es)
	SimulationModel::TriangleModelVector& tm = model->getTriangleModels();
	ParticleData& pd = model->getParticles();
	for (unsigned int i = 0; i < tm.size(); i++) {
		const unsigned int nVert = tm[i]->getParticleMesh().numVertices();
		unsigned int offset = tm[i]->getIndexOffset();
		tm[i]->setFrictionCoeff(static_cast<Real>(1));
		tm[i]->setThickness(i * 0.05);
		std::string name = "co_cloth_" + std::to_string(i);
		cd->addCollisionObjectWithoutGeometry(
			i, CollisionDetection::CollisionObject::TriangleModelCollisionObjectType,
			&pd.getPosition(offset), nVert, true, name);

		TriangleModel* tm1 = tm[i];
		const IndexedFaceMesh& mesh = tm1->getParticleMesh();

		auto sim = PBD::Simulation::getCurrent();
		auto* scd = dynamic_cast<PBD::CubicSDFCollisionDetection*>(
			sim->getTimeStep()->getCollisionDetection());
		CollisionDetection::CollisionObject* co = scd->getCollisionObjects().back();

		DistanceFieldCollisionDetection::DistanceFieldCollisionObject* dfco =
			dynamic_cast<DistanceFieldCollisionDetection::DistanceFieldCollisionObject*>(co);
	}
}

void render() { base->render(); }
