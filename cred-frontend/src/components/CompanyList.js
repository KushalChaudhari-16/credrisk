"use client"

import { useState, useEffect } from "react"
import { Card, Table, Spinner, Badge, Form, Row, Col } from "react-bootstrap"
import { FiTrendingUp, FiTrendingDown, FiMinus } from "react-icons/fi"
import api from "../services/api"

const CompanyList = ({ onSelectCompany }) => {
  const [companies, setCompanies] = useState([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState("all")
  const [sortBy, setSortBy] = useState("market_cap")

  useEffect(() => {
    fetchCompanies()
  }, [])

  const fetchCompanies = async () => {
    try {
      const response = await api.getCompanies()
      setCompanies(response.data.companies)
    } catch (error) {
      console.error("Error fetching companies:", error)
    } finally {
      setLoading(false)
    }
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

  const formatMarketCap = (marketCap) => {
    if (marketCap >= 1e12) return `$${(marketCap / 1e12).toFixed(1)}T`
    if (marketCap >= 1e9) return `$${(marketCap / 1e9).toFixed(1)}B`
    if (marketCap >= 1e6) return `$${(marketCap / 1e6).toFixed(1)}M`
    return `$${marketCap}`
  }

  const filteredAndSortedCompanies = companies
    .filter((company) => {
      if (filter === "all") return true
      if (filter === "scored") return company.latest_score !== null
      if (filter === "unscored") return company.latest_score === null
      return company.sector === filter
    })
    .sort((a, b) => {
      if (sortBy === "market_cap") return b.market_cap - a.market_cap
      if (sortBy === "score") return (b.latest_score || 0) - (a.latest_score || 0)
      if (sortBy === "symbol") return a.symbol.localeCompare(b.symbol)
      return 0
    })

  const sectors = [...new Set(companies.map((c) => c.sector))].filter((s) => s !== "Unknown")

  if (loading) {
    return (
      <div className="loading-spinner">
        <Spinner animation="border" variant="primary" />
      </div>
    )
  }

  return (
    <div>
      <div className="d-flex justify-content-between align-items-center mb-4">
        <h2>Company Portfolio</h2>
        <Badge bg="secondary">{filteredAndSortedCompanies.length} companies</Badge>
      </div>

      <Row className="mb-4">
        <Col md={6}>
          <Form.Select
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="bg-dark text-light border-secondary"
          >
            <option value="all">All Companies</option>
            <option value="scored">Scored Only</option>
            <option value="unscored">Unscored Only</option>
            {sectors.map((sector) => (
              <option key={sector} value={sector}>
                {sector}
              </option>
            ))}
          </Form.Select>
        </Col>
        <Col md={6}>
          <Form.Select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            className="bg-dark text-light border-secondary"
          >
            <option value="market_cap">Sort by Market Cap</option>
            <option value="score">Sort by Score</option>
            <option value="symbol">Sort by Symbol</option>
          </Form.Select>
        </Col>
      </Row>

      <Card className="card-dark">
        <Card.Body className="p-0">
          <Table responsive className="table-dark mb-0">
            <thead>
              <tr>
                <th>Company</th>
                <th>Sector</th>
                <th>Market Cap</th>
                <th>Credit Score</th>
                <th>Risk Grade</th>
                <th>Last Updated</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {filteredAndSortedCompanies.map((company) => (
                <tr key={company.symbol}>
                  <td>
                    <div>
                      <strong>{company.symbol}</strong>
                      <br />
                      <small className=" ">{company.name}</small>
                    </div>
                  </td>
                  <td>
                    <Badge bg="outline-secondary">{company.sector}</Badge>
                  </td>
                  <td>{formatMarketCap(company.market_cap)}</td>
                  <td>
                    {company.latest_score ? (
                      <div className="d-flex align-items-center">
                        <span>{company.latest_score.toFixed(2)}</span>
                        {company.latest_score >= 8 ? (
                          <FiTrendingUp className="text-success ms-1" />
                        ) : company.latest_score >= 6 ? (
                          <FiMinus className="text-warning ms-1" />
                        ) : (
                          <FiTrendingDown className="text-danger ms-1" />
                        )}
                      </div>
                    ) : (
                      <span className=" ">Not Rated</span>
                    )}
                  </td>
                  <td>
                    <span className={getRiskGradeClass(company.risk_grade)}>{company.risk_grade}</span>
                  </td>
                  <td>
                    {company.last_scored ? (
                      <small>{new Date(company.last_scored).toLocaleDateString()}</small>
                    ) : (
                      <small className=" ">Never</small>
                    )}
                  </td>
                  <td>
                    <button
                      className="btn btn-sm btn-outline-primary"
                      onClick={() => onSelectCompany({ symbol: company.symbol, name: company.name })}
                    >
                      Analyze
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </Table>
        </Card.Body>
      </Card>
    </div>
  )
}

export default CompanyList
