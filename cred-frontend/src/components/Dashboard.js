"use client"

import { useState, useEffect } from "react"
import { Card, Row, Col, Spinner } from "react-bootstrap"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from "recharts"
import { FiTrendingUp, FiAlertTriangle, FiGrid, FiActivity } from "react-icons/fi"
import api from "../services/api"

const Dashboard = () => {
  const [overview, setOverview] = useState(null)
  const [alerts, setAlerts] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchDashboardData()
  }, [])

  const fetchDashboardData = async () => {
    try {
      const [overviewRes, alertsRes] = await Promise.all([api.getMarketOverview(), api.getAlerts()])
      setOverview(overviewRes.data)
      setAlerts(alertsRes.data.active_alerts.slice(0, 5))
    } catch (error) {
      console.error("Error fetching dashboard data:", error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="loading-spinner">
        <Spinner animation="border" variant="primary" />
      </div>
    )
  }

  const COLORS = ["#58a6ff", "#2ea043", "#fb8500", "#f85149", "#7c3aed"]

  return (
    <div>
      <div className="d-flex justify-content-between align-items-center mb-4">
        <h2>Credit Intelligence Dashboard</h2>
        <small className=" ">Last updated: {new Date().toLocaleString()}</small>
      </div>

      <Row className="mb-4">
        <Col md={3}>
          <div className="metric-card">
            <FiGrid size={24} style={{ marginBottom: "10px" }} />
            <div className="metric-value">{overview?.market_overview?.total_companies_tracked || 0}</div>
            <div className="metric-label">Companies Tracked</div>
          </div>
        </Col>
        <Col md={3}>
          <div className="metric-card">
            <FiTrendingUp size={24} style={{ marginBottom: "10px" }} />
            <div className="metric-value">{overview?.market_overview?.market_average_score?.toFixed(1) || 0}</div>
            <div className="metric-label">Market Avg Score</div>
          </div>
        </Col>
        <Col md={3}>
          <div className="metric-card">
            <FiActivity size={24} style={{ marginBottom: "10px" }} />
            <div className="metric-value">{overview?.market_overview?.daily_score_updates || 0}</div>
            <div className="metric-label">Daily Updates</div>
          </div>
        </Col>
        <Col md={3}>
          <div className="metric-card">
            <FiAlertTriangle size={24} style={{ marginBottom: "10px" }} />
            <div className="metric-value">{alerts.length}</div>
            <div className="metric-label">Active Alerts</div>
          </div>
        </Col>
      </Row>

      <Row>
        <Col md={8}>
          <Card className="card-dark mb-4">
            <Card.Header>Sector Performance Analysis</Card.Header>
            <Card.Body>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={overview?.sector_analysis || []}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
                  <XAxis dataKey="sector" stroke="#c9d1d9" />
                  <YAxis stroke="#c9d1d9" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#161b22",
                      border: "1px solid #30363d",
                      color: "#c9d1d9",
                    }}
                  />
                  <Bar dataKey="average_score" fill="#58a6ff" />
                </BarChart>
              </ResponsiveContainer>
            </Card.Body>
          </Card>
        </Col>
        <Col md={4}>
          <Card className="card-dark mb-4">
            <Card.Header>Sector Distribution</Card.Header>
            <Card.Body>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={overview?.sector_analysis || []}
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="company_count"
                    label={({ sector, company_count }) => `${sector}: ${company_count}`}
                  >
                    {(overview?.sector_analysis || []).map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#161b22",
                      border: "1px solid #30363d",
                      color: "#c9d1d9",
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </Card.Body>
          </Card>
        </Col>
      </Row>

      <Row>
        <Col md={12}>
          <Card className="card-dark">
            <Card.Header>Recent Alerts</Card.Header>
            <Card.Body>
              {alerts.length > 0 ? (
                <div>
                  {alerts.map((alert, index) => (
                    <div key={index} className={`alert alert-${alert.severity === "high" ? "danger" : "warning"} mb-2`}>
                      <div className="d-flex justify-content-between align-items-center">
                        <div>
                          <strong>{alert.symbol}</strong> - {alert.company_name}
                          <br />
                          <small>
                            Score changed from {alert.previous_score?.toFixed(2)} to {alert.current_score?.toFixed(2)}(
                            {alert.score_change > 0 ? "+" : ""}
                            {alert.score_change?.toFixed(2)})
                          </small>
                        </div>
                        <div className="text-end">
                          <div className="badge bg-secondary">{alert.alert_type.replace("_", " ").toUpperCase()}</div>
                          <br />
                          <small>{new Date(alert.timestamp).toLocaleString()}</small>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className=" ">No active alerts</p>
              )}
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default Dashboard