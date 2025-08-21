"use client"

import { useState } from "react"
import "bootstrap/dist/css/bootstrap.min.css"
import "./App.css"
import Sidebar from "./components/Sidebar"
import Dashboard from "./components/Dashboard"
import CompanyList from "./components/CompanyList"
import CompanyDetails from "./components/CompanyDetails"
import CompanyHistory from "./components/CompanyHistory"
import CompanyComparison from "./components/CompanyComparison"
import AlertsPanel from "./components/AlertsPanel"
import MarketOverview from "./components/MarketOverview"
import CompanySearch from "./components/CompanySearch"

function App() {
  const [activeView, setActiveView] = useState("dashboard")
  const [selectedCompany, setSelectedCompany] = useState(null)

  const renderContent = () => {
    switch (activeView) {
      case "dashboard":
        return <Dashboard />
      case "companies":
        return <CompanyList onSelectCompany={setSelectedCompany} />
      case "company-details":
        return selectedCompany ? (
          <CompanyDetails company={selectedCompany} />
        ) : (
          <CompanySearch onSelectCompany={setSelectedCompany} />
        )
      case "company-history":
        return selectedCompany ? (
          <CompanyHistory company={selectedCompany} />
        ) : (
          <CompanySearch onSelectCompany={setSelectedCompany} />
        )
      case "company-comparison":
        return selectedCompany ? (
          <CompanyComparison company={selectedCompany} />
        ) : (
          <CompanySearch onSelectCompany={setSelectedCompany} />
        )
      case "alerts":
        return <AlertsPanel />
      case "market-overview":
        return <MarketOverview />
      case "search":
        return <CompanySearch onSelectCompany={setSelectedCompany} />
      default:
        return <Dashboard />
    }
  }

  return (
    <div className="App">
      <Sidebar activeView={activeView} setActiveView={setActiveView} selectedCompany={selectedCompany} />
      <div className="main-content">{renderContent()}</div>
    </div>
  )
}

export default App
