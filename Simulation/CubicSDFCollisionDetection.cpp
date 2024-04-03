#include "CubicSDFCollisionDetection.h"
#include "Simulation/IDFactory.h"
#include "Eigen/Dense"

using namespace PBD;

int CubicSDFCollisionDetection::CubicSDFCollisionObject::TYPE_ID = IDFactory::getId();

CubicSDFCollisionDetection::CubicSDFCollisionDetection() :
	DistanceFieldCollisionDetection()
{	
}

CubicSDFCollisionDetection::~CubicSDFCollisionDetection()
{
}

bool CubicSDFCollisionDetection::isDistanceFieldCollisionObject(CollisionObject *co) const
{
 	return DistanceFieldCollisionDetection::isDistanceFieldCollisionObject(co) ||
		(co->getTypeId() == CubicSDFCollisionDetection::CubicSDFCollisionObject::TYPE_ID);
}

void CubicSDFCollisionDetection::addCubicSDFCollisionObject(const unsigned int bodyIndex, const unsigned int bodyType, const Vector3r *vertices, const unsigned int numVertices, const std::string &sdfFile, const Vector3r &scale, const bool testMesh, const bool invertSDF)
{
	CubicSDFCollisionDetection::CubicSDFCollisionObject *co = new CubicSDFCollisionDetection::CubicSDFCollisionObject();
	co->m_bodyIndex = bodyIndex;
	co->m_bodyType = bodyType;
	co->m_sdfFile = sdfFile;
	co->m_scale = scale;
	co->m_sdf = std::make_shared<Grid>(co->m_sdfFile);
 	co->m_bvh.init(vertices, numVertices);
 	co->m_bvh.construct();
	co->m_testMesh = testMesh;
	co->m_invertSDF = 1.0;
	if (invertSDF)
		co->m_invertSDF = -1.0;
	m_collisionObjects.push_back(co);
}

void PBD::CubicSDFCollisionDetection::addCubicSDFCollisionObject(const unsigned int bodyIndex, const unsigned int bodyType, const Vector3r *vertices, const unsigned int numVertices, GridPtr sdf, const Vector3r &scale, const bool testMesh /*= true*/, const bool invertSDF /*= false*/, const std::string& name)
{
	CubicSDFCollisionDetection::CubicSDFCollisionObject *co = new CubicSDFCollisionDetection::CubicSDFCollisionObject();
	co->m_bodyIndex = bodyIndex;
	co->m_bodyType = bodyType;
	co->m_sdfFile = "";
	co->m_scale = scale;
	co->m_sdf = sdf;
	co->m_bvh.init(vertices, numVertices);
	co->m_bvh.construct();
	co->m_testMesh = testMesh;
	co->m_invertSDF = 1.0;
	if (invertSDF)
		co->m_invertSDF = -1.0;
	co->m_name = name;
	m_collisionObjects.push_back(co);
}

CubicSDFCollisionDetection::CubicSDFCollisionObject::CubicSDFCollisionObject()
{
}

CubicSDFCollisionDetection::CubicSDFCollisionObject::~CubicSDFCollisionObject()
{
}

double CubicSDFCollisionDetection::CubicSDFCollisionObject::distance(const Eigen::Vector3d &x, const Real tolerance)
{
	const Eigen::Vector3d scaled_x = x.cwiseProduct(m_scale.template cast<double>().cwiseInverse());
	const double dist = m_sdf->interpolate(0, scaled_x);
	if (dist == std::numeric_limits<double>::max())
		return dist;
	return m_invertSDF * m_scale[0]*dist - tolerance;
}

bool CubicSDFCollisionDetection::CubicSDFCollisionObject::collisionTest(const Vector3r &x, const Real tolerance, Vector3r &cp, Vector3r &n, Real &dist, const Real maxDist)
{
	const Vector3r scaled_x = x.cwiseProduct(m_scale.cwiseInverse());

	Eigen::Vector3d normal;	
	double d = m_sdf->interpolate(0, scaled_x.template cast<double>(), &normal);
	if (d == std::numeric_limits<Real>::max())
		return false;
	dist = static_cast<Real>(m_invertSDF * d - tolerance);

	normal = m_invertSDF * normal;
	if (dist < maxDist)
	{
		normal.normalize();
		n = normal.template cast<Real>();

		cp = (scaled_x - dist * n);
		cp = cp.cwiseProduct(m_scale);

		return true;
	}
	return false;
}

bool CubicSDFCollisionDetection::CubicSDFCollisionObject::collisionTestBatch(const std::vector<Vector3r> &x_batch, const Real tolerance, std::vector<Vector3r> &cp, std::vector<Vector3r> &n, std::vector<Real> &dist, const Real maxDist)
{
    bool collision = false;
    const int size = x_batch.size();
    cp.resize(size);
    n.resize(size);
    dist.resize(size);

    #pragma omp parallel for
    for (int i = 0; i < size; ++i)
    {
        const Vector3r &x = x_batch[i];
        // Vector3r &cp_single = cp[i];
        Vector3r &n_single = n[i];
        Real &dist_single = dist[i];

        const Vector3r scaled_x = x.cwiseProduct(m_scale.cwiseInverse());

        Eigen::Vector3d normal;    
        double d = m_sdf->interpolate(0, scaled_x.template cast<double>(), &normal);
        if (d == std::numeric_limits<Real>::max())
        {
            dist_single = std::numeric_limits<Real>::infinity();
            n_single = Vector3r(0, 0, 0);
        } else {
            dist_single = static_cast<Real>(m_invertSDF * d - tolerance);
        }

        normal = m_invertSDF * normal;
        if (dist_single < maxDist)
        {
            normal.normalize();
            n_single = normal.template cast<Real>();

            // cp_single = (scaled_x - dist_single * n_single);
            // cp_single = cp_single.cwiseProduct(m_scale);
        }
    }
    return collision;
}