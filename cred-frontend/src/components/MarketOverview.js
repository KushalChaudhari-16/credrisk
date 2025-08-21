"use client"

import { useState, useEffect } from "react"
import { Card, Row, Col, Spinner, Badge } from "react-bootstrap"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from "recharts"
import { FiTrendingUp, FiActivity, FiTarget, FiDatabase } from "react-icons/fi"
import api from "../services/api"

const MarketOverview = () => {
  const [overviewData, setOverviewData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchOverviewData()
  }, [])

  const fetchOverviewData = async () => {
    try {
      const response = await api.getMarketOverview()
      setOverviewData(response.data)
    } catch (error) {
      console.error("Error fetching market overview:", error)
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

  if (!overviewData) {
    return (
      <Card className="card-dark">
        <Card.Body>
          <p>Market overview data not available</p>
        </Card.Body>
      </Card>
    )
  }

  const COLORS = ["#58a6ff", "#2ea043", "#fb8500", "#f85149", "#7c3aed"]

  const modelPerformanceData = Object.entries(overviewData.system_metrics.model_performance).map(
    ([model, metrics]) => ({
      model: model.replace("_", " ").toUpperCase(),
      score: metrics.cv_score,
      mse: metrics.mse || 0,
      mae: metrics.mae || 0,
    }),
  )

  return (
    <div>
      <div className="d-flex justify-content-between align-items-center mb-4">
        <h2>Market Overview</h2>
        <small className=" ">Updated: {new Date(overviewData.timestamp).toLocaleString()}</small>
      </div>

      <Row className="mb-4">
        <Col md={3}>
          <Card className="card-dark text-center">
            <Card.Body>
              <FiDatabase size={24} className="text-primary mb-2" />
              <h3 className="mb-0" style={{ color: "#58a6ff" }}>
                {overviewData.market_overview.total_companies_tracked}
              </h3>
              <p className="mb-0">Companies</p>
              <small className=" ">Tracked</small>
            </Card.Body>
          </Card>
        </Col>
        <Col md={3}>
          <Card className="card-dark text-center">
            <Card.Body>
              <FiTrendingUp size={24} className="text-success mb-2" />
              <h3 className="mb-0" style={{ color: "#2ea043" }}>
                {overviewData.market_overview.market_average_score.toFixed(1)}
              </h3>
              <p className="mb-0">Market Average</p>
              <Badge className="grade-aa">{overviewData.market_overview.market_grade}</Badge>
            </Card.Body>
          </Card>
        </Col>
        <Col md={3}>
          <Card className="card-dark text-center">
            <Card.Body>
              <FiActivity size={24} className="text-warning mb-2" />
              <h3 className="mb-0" style={{ color: "#fb8500" }}>
                {overviewData.market_overview.daily_score_updates}
              </h3>
              <p className="mb-0">Daily Updates</p>
              <small className=" ">Score Refreshes</small>
            </Card.Body>
          </Card>
        </Col>
        <Col md={3}>
          <Card className="card-dark text-center">
            <Card.Body>
              <FiTarget size={24} className="text-info mb-2" />
              <h3 className="mb-0" style={{ color: "#7c3aed" }}>
                {overviewData.system_metrics.model_type.toUpperCase()}
              </h3>
              <p className="mb-0">Best Model</p>
              <small className=" ">Active Algorithm</small>
            </Card.Body>
          </Card>
        </Col>
      </Row>

      <Row className="mb-4">
        <Col md={8}>
          <Card className="card-dark">
            <Card.Header>Sector Performance Analysis</Card.Header>
            <Card.Body>
              <ResponsiveContainer width="100%" height={350}>
                <BarChart data={overviewData.sector_analysis}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
                  <XAxis dataKey="sector" stroke="#c9d1d9" angle={-45} textAnchor="end" height={100} />
                  <YAxis stroke="#c9d1d9" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#161b22",
                      border: "1px solid #30363d",
                      color: "#c9d1d9",
                    }}
                  />
                  <Bar dataKey="average_score" fill="#58a6ff" />
                  <Bar dataKey="company_count" fill="#2ea043" />
                </BarChart>
              </ResponsiveContainer>
            </Card.Body>
          </Card>
        </Col>
        <Col md={4}>
          <Card className="card-dark">
            <Card.Header>Sector Distribution</Card.Header>
            <Card.Body>
              <ResponsiveContainer width="100%" height={350}>
                <PieChart>
                  <Pie
                    data={overviewData.sector_analysis}
                    cx="50%"
                    cy="50%"
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="company_count"
                    label={({ sector, company_count }) => `${sector}: ${company_count}`}
                  >
                    {overviewData.sector_analysis.map((entry, index) => (
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
    </div>
  )
}

export default MarketOverview
