"use client"

import { useState, useEffect } from "react"
import { Card, Row, Col, Spinner, Badge, Alert } from "react-bootstrap"
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip } from "recharts"
import { FiAlertTriangle, FiTrendingDown, FiActivity } from "react-icons/fi"
import api from "../services/api"
import moment from "moment"

const AlertsPanel = () => {
  const [alertsData, setAlertsData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchAlerts()
  }, [])

  const fetchAlerts = async () => {
    try {
      const response = await api.getAlerts()
      setAlertsData(response.data)
    } catch (error) {
      console.error("Error fetching alerts:", error)
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

  if (!alertsData || alertsData.active_alerts.length === 0) {
    return (
      <div>
        <h2 className="mb-4">Active Alerts</h2>
        <Card className="card-dark">
          <Card.Body className="text-center py-5">
            <FiActivity size={48} className="  mb-3" />
            <h5>No Active Alerts</h5>
            <p className=" ">All companies are performing within normal parameters.</p>
          </Card.Body>
        </Card>
      </div>
    )
  }

  const severityData = [
    {
      name: "High",
      value: alertsData.high_severity_count,
      color: "#f85149",
    },
    {
      name: "Medium",
      value: alertsData.alert_count - alertsData.high_severity_count,
      color: "#fb8500",
    },
  ]

  const alertsByCompany = alertsData.active_alerts.reduce((acc, alert) => {
    const existing = acc.find((item) => item.symbol === alert.symbol)
    if (existing) {
      existing.count += 1
      existing.totalChange += Math.abs(alert.score_change)
    } else {
      acc.push({
        symbol: alert.symbol,
        name: alert.company_name,
        count: 1,
        totalChange: Math.abs(alert.score_change),
      })
    }
    return acc
  }, [])

  const getSeverityBadge = (severity) => {
    switch (severity) {
      case "high":
        return <Badge bg="danger">High</Badge>
      case "medium":
        return <Badge bg="warning">Medium</Badge>
      default:
        return <Badge bg="secondary">Low</Badge>
    }
  }

  const getSeverityIcon = (severity) => {
    switch (severity) {
      case "high":
        return <FiAlertTriangle className="text-danger" />
      case "medium":
        return <FiTrendingDown className="text-warning" />
      default:
        return <FiActivity className="text-secondary" />
    }
  }

  return (
    <div>
      <div className="d-flex justify-content-between align-items-center mb-4">
        <h2>Active Alerts</h2>
        <div>
          <Badge bg="danger" className="me-2">
            {alertsData.high_severity_count} High
          </Badge>
          <Badge bg="secondary">{alertsData.alert_count} Total</Badge>
        </div>
      </div>

      <Row className="mb-4">
        <Col md={3}>
          <Card className="card-dark text-center">
            <Card.Body>
              <FiAlertTriangle size={24} className="text-danger mb-2" />
              <h3 className="mb-0" style={{ color: "#f85149" }}>
                {alertsData.alert_count}
              </h3>
              <p className="mb-0">Total Alerts</p>
              <small className=" ">Active Now</small>
            </Card.Body>
          </Card>
        </Col>
        <Col md={3}>
          <Card className="card-dark text-center">
            <Card.Body>
              <FiTrendingDown size={24} className="text-danger mb-2" />
              <h3 className="mb-0" style={{ color: "#f85149" }}>
                {alertsData.high_severity_count}
              </h3>
              <p className="mb-0">High Severity</p>
              <small className=" ">Immediate Attention</small>
            </Card.Body>
          </Card>
        </Col>
        <Col md={3}>
          <Card className="card-dark text-center">
            <Card.Body>
              <FiActivity size={24} className="text-warning mb-2" />
              <h3 className="mb-0" style={{ color: "#fb8500" }}>
                {alertsByCompany.length}
              </h3>
              <p className="mb-0">Companies</p>
              <small className=" ">Affected</small>
            </Card.Body>
          </Card>
        </Col>
        <Col md={3}>
          <Card className="card-dark text-center">
            <Card.Body>
              <FiTrendingDown size={24} className="text-info mb-2" />
              <h3 className="mb-0" style={{ color: "#58a6ff" }}>
                {(
                  alertsData.active_alerts.reduce((sum, alert) => sum + Math.abs(alert.score_change), 0) /
                  alertsData.active_alerts.length
                ).toFixed(1)}
              </h3>
              <p className="mb-0">Avg Change</p>
              <small className=" ">Score Points</small>
            </Card.Body>
          </Card>
        </Col>
      </Row>

      <Row className="mb-4">
        <Col md={8}>
          <Card className="card-dark">
            <Card.Header>Alerts by Company</Card.Header>
            <Card.Body>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={alertsByCompany}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
                  <XAxis dataKey="symbol" stroke="#c9d1d9" />
                  <YAxis stroke="#c9d1d9" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#161b22",
                      border: "1px solid #30363d",
                      color: "#c9d1d9",
                    }}
                  />
                  <Bar dataKey="totalChange" fill="#f85149" />
                </BarChart>
              </ResponsiveContainer>
            </Card.Body>
          </Card>
        </Col>
        <Col md={4}>
          <Card className="card-dark">
            <Card.Header>Severity Distribution</Card.Header>
            <Card.Body>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={severityData}
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                    label={({ name, value }) => `${name}: ${value}`}
                  >
                    {severityData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
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
            <Card.Header>Alert Details</Card.Header>
            <Card.Body>
              {alertsData.active_alerts.map((alert, index) => (
                <Alert key={index} variant={alert.severity === "high" ? "danger" : "warning"} className="mb-3">
                  <div className="d-flex justify-content-between align-items-start">
                    <div className="d-flex align-items-start">
                      {getSeverityIcon(alert.severity)}
                      <div className="ms-3">
                        <div className="d-flex align-items-center mb-2">
                          <h6 className="mb-0 me-2">
                            {alert.symbol} - {alert.company_name}
                          </h6>
                          {getSeverityBadge(alert.severity)}
                        </div>
                        <div className="mb-2">
                          <strong>Score Change:</strong> {alert.previous_score?.toFixed(2)} →{" "}
                          {alert.current_score?.toFixed(2)}
                          <span className="text-danger ms-2">
                            ({alert.score_change > 0 ? "+" : ""}
                            {alert.score_change?.toFixed(2)})
                          </span>
                        </div>
                        <div className="mb-2">
                          <strong>Grade Change:</strong> {alert.previous_grade} → {alert.current_grade}
                        </div>
                        <div>
                          <strong>Alert Type:</strong> {alert.alert_type.replace("_", " ").toUpperCase()}
                        </div>
                      </div>
                    </div>
                    <div className="text-end">
                      <small className=" ">{moment(alert.timestamp).fromNow()}</small>
                      <br />
                      <small className=" ">{moment(alert.timestamp).format("MMM DD, YYYY HH:mm")}</small>
                    </div>
                  </div>
                </Alert>
              ))}
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default AlertsPanel
