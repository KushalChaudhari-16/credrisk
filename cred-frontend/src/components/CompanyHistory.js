"use client"

import { useState, useEffect } from "react"
import { Card, Row, Col, Spinner, Badge } from "react-bootstrap"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from "recharts"
import { FiTrendingUp, FiTrendingDown, FiActivity } from "react-icons/fi"
import api from "../services/api"
import moment from "moment"

const CompanyHistory = ({ company }) => {
  const [historyData, setHistoryData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (company) {
      fetchCompanyHistory()
    }
  }, [company])

  const fetchCompanyHistory = async () => {
    try {
      const response = await api.getCompanyHistory(company.symbol)
      setHistoryData(response.data)
    } catch (error) {
      console.error("Error fetching company history:", error)
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

  if (!historyData || !historyData.history.length) {
    return (
      <Card className="card-dark">
        <Card.Body>
          <p>No historical data available for {company.symbol}</p>
        </Card.Body>
      </Card>
    )
  }

  const chartData = historyData.history
    .sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp))
    .map((item) => ({
      date: moment(item.timestamp).format("MMM DD"),
      score: item.score,
      confidence: item.confidence * 10,
      timestamp: item.timestamp,
    }))

  const getTrendIcon = () => {
    if (historyData.analytics.trend === "improving") {
      return <FiTrendingUp className="text-success" />
    } else if (historyData.analytics.trend === "declining") {
      return <FiTrendingDown className="text-danger" />
    }
    return <FiActivity className="text-warning" />
  }

  const getTrendColor = () => {
    if (historyData.analytics.trend === "improving") return "#2ea043"
    if (historyData.analytics.trend === "declining") return "#f85149"
    return "#fb8500"
  }

  return (
    <div>
      <div className="d-flex justify-content-between align-items-center mb-4">
        <div>
          <h2>{company.symbol} - Score History</h2>
          <p className="  mb-0">{company.name}</p>
        </div>
        <Badge bg="secondary">{historyData.analytics.data_points} data points</Badge>
      </div>

      <Row className="mb-4">
        <Col md={3}>
          <Card className="card-dark text-center">
            <Card.Body>
              <h3 className="mb-0" style={{ color: "#58a6ff" }}>
                {historyData.analytics.average_score}
              </h3>
              <p className="mb-0">Average Score</p>
              <small className=" ">Last {historyData.analytics.days_analyzed} days</small>
            </Card.Body>
          </Card>
        </Col>
        <Col md={3}>
          <Card className="card-dark text-center">
            <Card.Body>
              <div className="d-flex justify-content-center align-items-center mb-2">
                {getTrendIcon()}
                <span className="ms-2" style={{ color: getTrendColor() }}>
                  {historyData.analytics.trend.toUpperCase()}
                </span>
              </div>
              <p className="mb-0">Trend</p>
              <small className=" ">Direction</small>
            </Card.Body>
          </Card>
        </Col>
        <Col md={3}>
          <Card className="card-dark text-center">
            <Card.Body>
              <h3 className="mb-0" style={{ color: "#fb8500" }}>
                {historyData.analytics.volatility.toFixed(2)}
              </h3>
              <p className="mb-0">Volatility</p>
              <small className=" ">Score Variance</small>
            </Card.Body>
          </Card>
        </Col>
        <Col md={3}>
          <Card className="card-dark text-center">
            <Card.Body>
              <h3 className="mb-0" style={{ color: "#7c3aed" }}>
                {historyData.analytics.score_range[1] - historyData.analytics.score_range[0] > 0
                  ? (historyData.analytics.score_range[1] - historyData.analytics.score_range[0]).toFixed(2)
                  : "0.00"}
              </h3>
              <p className="mb-0">Range</p>
              <small className=" ">Max - Min</small>
            </Card.Body>
          </Card>
        </Col>
      </Row>

      <Row className="mb-4">
        <Col md={12}>
          <Card className="card-dark">
            <Card.Header>Credit Score Trend</Card.Header>
            <Card.Body>
              <ResponsiveContainer width="100%" height={400}>
                <AreaChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
                  <XAxis dataKey="date" stroke="#c9d1d9" />
                  <YAxis stroke="#c9d1d9" domain={[0, 10]} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#161b22",
                      border: "1px solid #30363d",
                      color: "#c9d1d9",
                    }}
                    labelFormatter={(label, payload) => {
                      if (payload && payload[0]) {
                        return moment(payload[0].payload.timestamp).format("MMMM DD, YYYY HH:mm")
                      }
                      return label
                    }}
                  />
                  <Area
                    type="monotone"
                    dataKey="score"
                    stroke="#58a6ff"
                    fill="#58a6ff"
                    fillOpacity={0.3}
                    strokeWidth={2}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </Card.Body>
          </Card>
        </Col>
      </Row>

      <Row>
        <Col md={12}>
          <Card className="card-dark">
            <Card.Header>Score & Confidence History</Card.Header>
            <Card.Body>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
                  <XAxis dataKey="date" stroke="#c9d1d9" />
                  <YAxis stroke="#c9d1d9" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#161b22",
                      border: "1px solid #30363d",
                      color: "#c9d1d9",
                    }}
                  />
                  <Line type="monotone" dataKey="score" stroke="#58a6ff" strokeWidth={2} name="Credit Score" />
                  <Line type="monotone" dataKey="confidence" stroke="#2ea043" strokeWidth={2} name="Confidence (x10)" />
                </LineChart>
              </ResponsiveContainer>
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default CompanyHistory
