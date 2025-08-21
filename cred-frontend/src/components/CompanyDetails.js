"use client"

import { useState, useEffect } from "react"
import { Card, Row, Col, Spinner, Badge } from "react-bootstrap"
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts"
import { FiAlertTriangle, FiCheckCircle } from "react-icons/fi"
import api from "../services/api"

const CompanyDetails = ({ company }) => {
  const [companyData, setCompanyData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (company) {
      fetchCompanyDetails()
    }
  }, [company])

  const fetchCompanyDetails = async () => {
    try {
      const response = await api.getCompanyScore(company.symbol)
      setCompanyData(response.data)
    } catch (error) {
      console.error("Error fetching company details:", error)
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

  if (!companyData) {
    return (
      <Card className="card-dark">
        <Card.Body>
          <p>Company data not available for {company.symbol}</p>
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

  const radarData = companyData.feature_analysis.top_factors.slice(0, 6).map((factor) => ({
    factor: factor.factor.replace("_", " ").toUpperCase(),
    value: Math.abs(factor.impact) * 10,
    fullMark: 10,
  }))

  const factorData = companyData.explanation.feature_breakdown.strengths.factors
    .concat(companyData.explanation.feature_breakdown.weaknesses.factors)
    .slice(0, 8)
    .map((factor) => ({
      name: factor.factor.replace("_", " "),
      impact: factor.impact,
      importance: factor.importance * 100,
    }))

  return (
    <div>
      <div className="d-flex justify-content-between align-items-center mb-4">
        <div>
          <h2>
            {companyData.symbol} - {companyData.name}
          </h2>
          <Badge bg="secondary">{companyData.sector}</Badge>
        </div>
        <small className=" ">Last updated: {new Date(companyData.last_updated).toLocaleString()}</small>
      </div>

      <Row className="mb-4">
        <Col md={3}>
          <Card className="card-dark text-center">
            <Card.Body>
              <h1 className="display-4 mb-0" style={{ color: "#58a6ff" }}>
                {companyData.credit_score.toFixed(1)}
              </h1>
              <p className="mb-0">Credit Score</p>
              <small className=" ">out of 10</small>
            </Card.Body>
          </Card>
        </Col>
        <Col md={3}>
          <Card className="card-dark text-center">
            <Card.Body>
              <h2 className={`mb-0 ${getRiskGradeClass(companyData.risk_grade)}`}>{companyData.risk_grade}</h2>
              <p className="mb-0">Risk Grade</p>
              <small className=" ">Credit Rating</small>
            </Card.Body>
          </Card>
        </Col>
        <Col md={3}>
          <Card className="card-dark text-center">
            <Card.Body>
              <h2 className="mb-0" style={{ color: "#2ea043" }}>
                {(companyData.confidence * 100).toFixed(0)}%
              </h2>
              <p className="mb-0">Confidence</p>
              <small className=" ">Model Certainty</small>
            </Card.Body>
          </Card>
        </Col>
        <Col md={3}>
          <Card className="card-dark text-center">
            <Card.Body>
              <h2 className="mb-0" style={{ color: "#fb8500" }}>
                {companyData.model_info.best_model.toUpperCase()}
              </h2>
              <p className="mb-0">Best Model</p>
              <small className=" ">Algorithm Used</small>
            </Card.Body>
          </Card>
        </Col>
      </Row>

      <Row className="mb-4">
        <Col md={8}>
          <Card className="card-dark">
            <Card.Header>Executive Summary</Card.Header>
            <Card.Body>
              <p>{companyData.explanation.executive_summary}</p>

              <h6 className="mt-4">Key Risk Factors</h6>
              {companyData.explanation.risk_assessment.high_risk_factors.map((risk, index) => (
                <div key={index} className="alert alert-danger mb-2">
                  <div className="d-flex align-items-center">
                    <FiAlertTriangle className="me-2" />
                    <div>
                      <strong>{risk.factor}</strong>
                      <br />
                      <small>{risk.description}</small>
                    </div>
                  </div>
                </div>
              ))}

              {companyData.explanation.risk_assessment.medium_risk_factors.map((risk, index) => (
                <div key={index} className="alert alert-warning mb-2">
                  <div className="d-flex align-items-center">
                    <FiAlertTriangle className="me-2" />
                    <div>
                      <strong>{risk.factor}</strong>
                      <br />
                      <small>{risk.description}</small>
                    </div>
                  </div>
                </div>
              ))}
            </Card.Body>
          </Card>
        </Col>
        <Col md={4}>
          <Card className="card-dark">
            <Card.Header>Factor Analysis</Card.Header>
            <Card.Body>
              <ResponsiveContainer width="100%" height={250}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="#30363d" />
                  <PolarAngleAxis dataKey="factor" tick={{ fontSize: 10, fill: "#c9d1d9" }} />
                  <PolarRadiusAxis tick={{ fontSize: 8, fill: "#7d8590" }} />
                  <Radar name="Impact" dataKey="value" stroke="#58a6ff" fill="#58a6ff" fillOpacity={0.3} />
                </RadarChart>
              </ResponsiveContainer>
            </Card.Body>
          </Card>
        </Col>
      </Row>

      <Row className="mb-4">
        <Col md={12}>
          <Card className="card-dark">
            <Card.Header>Feature Impact Analysis</Card.Header>
            <Card.Body>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={factorData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
                  <XAxis dataKey="name" stroke="#c9d1d9" angle={-45} textAnchor="end" height={100} />
                  <YAxis stroke="#c9d1d9" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#161b22",
                      border: "1px solid #30363d",
                      color: "#c9d1d9",
                    }}
                  />
                  <Bar dataKey="impact" fill={(entry) => (entry.impact > 0 ? "#2ea043" : "#f85149")} />
                </BarChart>
              </ResponsiveContainer>
            </Card.Body>
          </Card>
        </Col>
      </Row>

      <Row>
        <Col md={6}>
          <Card className="card-dark">
            <Card.Header>Strengths</Card.Header>
            <Card.Body>
              {companyData.explanation.feature_breakdown.strengths.factors.map((factor, index) => (
                <div
                  key={index}
                  className="d-flex justify-content-between align-items-center mb-2 p-2 bg-success bg-opacity-10 rounded"
                >
                  <div>
                    <FiCheckCircle className="text-success me-2" />
                    <strong>{factor.factor.replace("_", " ").toUpperCase()}</strong>
                    <br />
                    <small className=" ">{factor.description}</small>
                  </div>
                  <Badge bg="success">+{factor.impact.toFixed(2)}</Badge>
                </div>
              ))}
            </Card.Body>
          </Card>
        </Col>
        <Col md={6}>
          <Card className="card-dark">
            <Card.Header>Weaknesses</Card.Header>
            <Card.Body>
              {companyData.explanation.feature_breakdown.weaknesses.factors.map((factor, index) => (
                <div
                  key={index}
                  className="d-flex justify-content-between align-items-center mb-2 p-2 bg-danger bg-opacity-10 rounded"
                >
                  <div>
                    <FiAlertTriangle className="text-danger me-2" />
                    <strong>{factor.factor.replace("_", " ").toUpperCase()}</strong>
                    <br />
                    <small className=" ">{factor.description}</small>
                  </div>
                  <Badge bg="danger">{factor.impact.toFixed(2)}</Badge>
                </div>
              ))}
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default CompanyDetails
