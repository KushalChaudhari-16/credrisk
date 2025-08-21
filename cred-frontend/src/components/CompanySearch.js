"use client"

import { useState } from "react"
import { Card, Form, Spinner } from "react-bootstrap"
import { FiSearch } from "react-icons/fi"
import api from "../services/api"

const CompanySearch = ({ onSelectCompany }) => {
  const [searchTerm, setSearchTerm] = useState("")
  const [searchResults, setSearchResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [showResults, setShowResults] = useState(false)

  const handleSearch = async (term) => {
    if (term.length < 2) {
      setSearchResults([])
      setShowResults(false)
      return
    }

    setLoading(true)
    try {
      const response = await api.searchCompanies(term)
      setSearchResults(response.data.quotes || [])
      setShowResults(true)
    } catch (error) {
      console.error("Error searching companies:", error)
      setSearchResults([])
    } finally {
      setLoading(false)
    }
  }

  const handleInputChange = (e) => {
    const value = e.target.value
    setSearchTerm(value)
    handleSearch(value)
  }

  const handleSelectCompany = (company) => {
    onSelectCompany({
      symbol: company.symbol,
      name: company.longname || company.shortname,
    })
    setSearchTerm("")
    setShowResults(false)
  }

  return (
    <div>
      <div className="d-flex justify-content-between align-items-center mb-4">
        <h2>Search Companies</h2>
      </div>

      <Card className="card-dark">
        <Card.Body>
          <div className="search-container">
            <div className="input-group">
              <span className="input-group-text bg-dark border-secondary">
                <FiSearch />
              </span>
              <Form.Control
                type="text"
                placeholder="Search for companies (e.g., Apple, Microsoft, Tesla...)"
                value={searchTerm}
                onChange={handleInputChange}
                className="bg-dark text-light border-secondary"
                autoComplete="off"
              />
            </div>

            {showResults && (
              <div className="search-dropdown">
                {loading ? (
                  <div className="search-item text-center">
                    <Spinner animation="border" size="sm" />
                    <span className="ms-2">Searching...</span>
                  </div>
                ) : searchResults.length > 0 ? (
                  searchResults.slice(0, 10).map((company, index) => (
                    <div key={index} className="search-item" onClick={() => handleSelectCompany(company)}>
                      <div>
                        <strong>{company.symbol}</strong>
                        {company.exchDisp && <span className="badge bg-secondary ms-2 small">{company.exchDisp}</span>}
                      </div>
                      <div className="small  ">{company.longname || company.shortname}</div>
                      {company.sectorDisp && <div className="small  ">{company.sectorDisp}</div>}
                    </div>
                  ))
                ) : (
                  <div className="search-item  ">No companies found for "{searchTerm}"</div>
                )}
              </div>
            )}
          </div>

          <div className="mt-4">
            <h5>How to use:</h5>
            <ul className=" ">
              <li>Type at least 2 characters to start searching</li>
              <li>Search by company name or stock symbol</li>
              <li>Click on a result to select and analyze the company</li>
              <li>Popular examples: AAPL, MSFT, GOOGL, TSLA, AMZN</li>
            </ul>
          </div>
        </Card.Body>
      </Card>
    </div>
  )
}

export default CompanySearch
