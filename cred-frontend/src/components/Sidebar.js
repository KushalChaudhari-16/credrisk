"use client"

import {
  FiHome,
  FiGrid,
  FiTrendingUp,
  FiAlertTriangle,
  FiPieChart,
  FiSearch,
  FiGitMerge,
  FiClock,
} from "react-icons/fi"

const Sidebar = ({ activeView, setActiveView, selectedCompany }) => {
  const navItems = [
    {
      group: "Overview",
      items: [
        { id: "dashboard", label: "Dashboard", icon: FiHome },
        { id: "market-overview", label: "Market Overview", icon: FiPieChart },
        { id: "alerts", label: "Active Alerts", icon: FiAlertTriangle },
      ],
    },
    {
      group: "Companies",
      items: [
        { id: "search", label: "Search Companies", icon: FiSearch },
        { id: "companies", label: "All Companies", icon: FiGrid },
      ],
    },
    {
      group: "Analysis",
      items: [
        { id: "company-details", label: "Company Score", icon: FiTrendingUp },
        { id: "company-history", label: "Score History", icon: FiClock },
        { id: "company-comparison", label: "Agency Comparison", icon: FiGitMerge },
      ],
    },
  ]

  return (
    <div className="sidebar">
      <div className="brand">
        <h3>CredTech AI</h3>
        <p>Real-Time Credit Intelligence</p>
      </div>

      {selectedCompany && (
        <div style={{ padding: "15px 20px", borderBottom: "1px solid #30363d", backgroundColor: "#21262d" }}>
          <div style={{ fontSize: "12px", color: "#7d8590", marginBottom: "5px" }}>SELECTED COMPANY</div>
          <div style={{ fontWeight: "600", color: "#58a6ff" }}>{selectedCompany.symbol}</div>
          <div style={{ fontSize: "12px", color: "#c9d1d9" }}>{selectedCompany.name}</div>
        </div>
      )}

      {navItems.map((group, groupIndex) => (
        <div key={groupIndex} className="nav-group">
          <div className="nav-group-title">{group.group}</div>
          {group.items.map((item) => {
            const Icon = item.icon
            return (
              <div
                key={item.id}
                className={`nav-item ${activeView === item.id ? "active" : ""}`}
                onClick={() => setActiveView(item.id)}
              >
                <Icon size={18} />
                <span>{item.label}</span>
              </div>
            )
          })}
        </div>
      ))}
    </div>
  )
}

export default Sidebar