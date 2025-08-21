"use client"

import { useState, useEffect } from "react"
import { Card, Row, Col, Spinner, Badge, ProgressBar } from "react-bootstrap"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadialBarChart,
  RadialBar,
} from "recharts"
import { FiTrendingUp, FiTrendingDown, FiTarget } from "react-icons/fi"
import api from "../services/api"

const CompanyComparison = ({ company }) => {
  const [comparisonData, setComparisonData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (company) {
      fetchComparisonData()
    }
  }, [company])

  const fetchComparisonData = async () => {
    try {
      const response = await api.getCompanyComparison(company.symbol, "sp")
      setComparisonData(response.data)
    } catch (error) {
      console.error("Error fetching comparison data:", error)
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

  if (!comparisonData) {
    return (
      <Card className="card-dark">
        <Card.Body>
          <p>Comparison data not available for {company.symbol}</p>
        </Card.Body>
      </Card>
    )
  }

  const getRiskGradeClass = (grade) => {
    if (grade.startsWith("AAA")) return "grade-aaa"
    if (grade.startsWith("AA")) return "grade-aa"
    if (grade.startsWith("A")) return "grade-a"
    if (grade.startsWith("BBB")) return "grade-bbb"
    if (grade.startsWith("BB")) return "grade-bb"
    if (grade.startsWith("B")) return "grade-b"
    if (grade.startsWith("CCC")) return "grade-ccc"
    return ""
  }

  const chartData = [
    {
      name: "CredTech AI",
      score: comparisonData.comparison.credtech_score,
      grade: comparisonData.comparison.credtech_grade,
    },
    {
      name: "S&P Rating",
      score: comparisonData.comparison.agency_score,
      grade: comparisonData.comparison.agency_grade,
    },
  ]

  const radialData = [
    {
      name: "CredTech",
      value: comparisonData.comparison.credtech_score,
      fill: "#58a6ff",
    },
    {
      name: "S&P",
      value: comparisonData.comparison.agency_score,
      fill: "#fb8500",
    },
  ]

  const getArbitrageIcon = () => {
    if (comparisonData.analysis.arbitrage_opportunity) {
      return <FiTrendingUp className="text-success" />
    }
    return <FiTarget className="text-warning" />
  }

  const getArbitrageColor = () => {
    if (comparisonData.analysis.arbitrage_opportunity) return "#2ea043"
    return "#fb8500"
  }

  return (
    <div>
      <div className="d-flex justify-content-between align-items-center mb-4">
        <div>
          <h2>{company.symbol} - Agency Comparison</h2>
          <p className="  mb-0">{company.name}</p>
        </div>
        <small className=" ">Updated: {new Date(comparisonData.timestamp).toLocaleString()}</small>
      </div>

      <Row className="mb-4">
        <Col md={3}>
          <Card className="card-dark text-center">
            <Card.Body>
              <h3 className="mb-0" style={{ color: "#58a6ff" }}>
                {comparisonData.comparison.credtech_score.toFixed(1)}
              </h3>
              <p className="mb-0">CredTech Score</p>
              <Badge className={getRiskGradeClass(comparisonData.comparison.credtech_grade)}>
                {comparisonData.comparison.credtech_grade}
              </Badge>
            </Card.Body>
          </Card>
        </Col>
        <Col md={3}>
          <Card className="card-dark text-center">
            <Card.Body>
              <h3 className="mb-0" style={{ color: "#fb8500" }}>
                {comparisonData.comparison.agency_score.toFixed(1)}
              </h3>
              <p className="mb-0">S&P Score</p>
              <Badge className={getRiskGradeClass(comparisonData.comparison.agency_grade)}>
                {comparisonData.comparison.agency_grade}
              </Badge>
            </Card.Body>
          </Card>
        </Col>
        <Col md={3}>
          <Card className="card-dark text-center">
            <Card.Body>
              <div className="d-flex justify-content-center align-items-center mb-2">
                {comparisonData.comparison.score_difference > 0 ? (
                  <FiTrendingUp className="text-success" />
                ) : (
                  <FiTrendingDown className="text-danger" />
                )}
                <span
                  className="ms-2"
                  style={{
                    color: comparisonData.comparison.score_difference > 0 ? "#2ea043" : "#f85149",
                  }}
                >
                  {comparisonData.comparison.score_difference > 0 ? "+" : ""}
                  {comparisonData.comparison.score_difference.toFixed(1)}
                </span>
              </div>
              <p className="mb-0">Difference</p>
              <small className=" ">CredTech vs S&P</small>
            </Card.Body>
          </Card>
        </Col>
        <Col md={3}>
          <Card className="card-dark text-center">
            <Card.Body>
              <div className="d-flex justify-content-center align-items-center mb-2">
                {getArbitrageIcon()}
                <span className="ms-2" style={{ color: getArbitrageColor() }}>
                  {comparisonData.analysis.opportunity_type.replace("_", " ").toUpperCase()}
                </span>
              </div>
              <p className="mb-0">Opportunity</p>
              <small className=" ">Market Assessment</small>
            </Card.Body>
          </Card>
        </Col>
      </Row>

      <Row className="mb-4">
        <Col md={8}>
          <Card className="card-dark">
            <Card.Header>Score Comparison</Card.Header>
            <Card.Body>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
                  <XAxis dataKey="name" stroke="#c9d1d9" />
                  <YAxis stroke="#c9d1d9" domain={[0, 10]} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#161b22",
                      border: "1px solid #30363d",
                      color: "#c9d1d9",
                    }}
                  />
                  <Bar dataKey="score" fill="#58a6ff" />
                </BarChart>
              </ResponsiveContainer>
            </Card.Body>
          </Card>
        </Col>
        <Col md={4}>
          <Card className="card-dark">
            <Card.Header>Radial Comparison</Card.Header>
            <Card.Body>
              <ResponsiveContainer width="100%" height={300}>
                <RadialBarChart cx="50%" cy="50%" innerRadius="20%" outerRadius="80%" data={radialData}>
                  <RadialBar dataKey="value" cornerRadius={10} fill="#8884d8" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#161b22",
                      border: "1px solid #30363d",
                      color: "#c9d1d9",
                    }}
                  />
                </RadialBarChart>
              </ResponsiveContainer>
            </Card.Body>
          </Card>
        </Col>
      </Row>

      <Row>
        <Col md={12}>
          <Card className="card-dark">
            <Card.Header>Analysis Summary</Card.Header>
            <Card.Body>
              <Row>
                <Col md={6}>
                  <h6>Arbitrage Analysis</h6>
                  <div
                    className={`alert ${comparisonData.analysis.arbitrage_opportunity ? "alert-success" : "alert-warning"}`}
                  >
                    <div className="d-flex align-items-center">
                      {getArbitrageIcon()}
                      <div className="ms-2">
                        <strong>
                          {comparisonData.analysis.arbitrage_opportunity ? "Opportunity Detected" : "Fairly Valued"}
                        </strong>
                        <br />
                        <small>
                          {comparisonData.analysis.arbitrage_opportunity
                            ? `Potential alpha: ${comparisonData.analysis.potential_alpha}%`
                            : "No significant arbitrage opportunity detected"}
                        </small>
                      </div>
                    </div>
                  </div>
                </Col>
                <Col md={6}>
                  <h6>Confidence Assessment</h6>
                  <div className="mb-3">
                    <div className="d-flex justify-content-between">
                      <span>CredTech Confidence</span>
                      <span>{(comparisonData.comparison.credtech_confidence * 100).toFixed(0)}%</span>
                    </div>
                    <ProgressBar now={comparisonData.comparison.credtech_confidence * 100} variant="info" />
                  </div>
                  <div className="mb-3">
                    <div className="d-flex justify-content-between">
                      <span>Agreement Level</span>
                      <span>{comparisonData.analysis.confidence_in_difference}</span>
                    </div>
                    <ProgressBar
                      now={
                        comparisonData.analysis.confidence_in_difference === "high"
                          ? 80
                          : comparisonData.analysis.confidence_in_difference === "medium"
                            ? 60
                            : 40
                      }
                      variant={
                        comparisonData.analysis.confidence_in_difference === "high"
                          ? "success"
                          : comparisonData.analysis.confidence_in_difference === "medium"
                            ? "warning"
                            : "danger"
                      }
                    />
                  </div>
                </Col>
              </Row>
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default CompanyComparison
