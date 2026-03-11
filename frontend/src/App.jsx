import { useState, useCallback, useEffect } from 'react'

const API = '/api'

const STEPS = [
  { label: 'Company Profile', section: 'DATA COLLECTION' },
  { label: 'Document Upload', section: 'DATA COLLECTION' },
  { label: 'Document Review', section: 'CLASSIFICATION' },
  { label: 'Schema Config', section: 'SCHEMA' },
  { label: 'GST & Banking', section: 'STRUCTURED DATA' },
  { label: 'Field Insights', section: 'QUALITATIVE' },
  { label: 'Financial Data', section: 'FINANCIAL' },
  { label: 'Run Analysis', section: 'ANALYSIS' },
  { label: 'Credit Decision', section: 'OUTPUT' },
]

const LOADING_PHASES = [
  'Computing financial ratios',
  'Running secondary research',
  'Constructing fraud network graph',
  'Forecasting cashflows (Prophet)',
  'Aggregating risk signals',
  'Running ML ensemble scoring',
  'Generating SHAP explanations',
  'Convening AI credit committee',
  'Generating SWOT analysis',
  'Generating GenAI narrative',
  'Generating Credit Appraisal Memo',
]

const Input = ({ label, value, onChange, type = 'text', opts = {} }) => (
  <div className="form-group">
    <label>{label}</label>
    {opts.select ? (
      <select value={value} onChange={e => onChange(e.target.value)}>
        {opts.options.map(o => <option key={o} value={o}>{o}</option>)}
      </select>
    ) : opts.textarea ? (
      <textarea value={value} onChange={e => onChange(e.target.value)} placeholder={opts.placeholder || ''} />
    ) : (
      <input type={type} value={value} onChange={e => onChange(e.target.value)} placeholder={opts.placeholder || ''} />
    )}
  </div>
)

export default function App() {
  const [step, setStep] = useState(0)
  const [loading, setLoading] = useState(false)
  const [loadingPhase, setLoadingPhase] = useState(0)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [activeTab, setActiveTab] = useState('summary')
  const [files, setFiles] = useState([])

  // Classification state
  const [classifications, setClassifications] = useState([])
  const [docTypes, setDocTypes] = useState([])
  const [classifying, setClassifying] = useState(false)

  // Schema state
  const [schemas, setSchemas] = useState({})
  const [activeSchemaType, setActiveSchemaType] = useState('')

  // PDF download state
  const [downloading, setDownloading] = useState(false)

  const [company, setCompany] = useState({
    company_name: '', industry: '', cin: '', years_in_business: 5,
    promoter_names: '', promoter_experience_yrs: 10, cibil_score: 700,
    collateral_type: 'immovable', security_coverage: 1.3,
    sector_outlook: 'stable', macro_risk_score: 5,
  })

  const [financials, setFinancials] = useState({
    revenue_cr: 10, ebitda_cr: 1.5, pat_cr: 0.8, total_debt_cr: 5,
    net_worth_cr: 4, interest_expense_cr: 0.5, depreciation_cr: 0.3,
    current_assets_cr: 6, current_liabilities_cr: 4, inventory_cr: 2,
    fixed_assets_cr: 8, annual_debt_service_cr: 1.2,
    revenue_history: '7,8,9,10', promoter_contribution_pct: 55,
  })

  const [gst, setGst] = useState({ gstr_2a_itc: 0, gstr_3b_itc: 0, gst_turnover: 0 })
  const [bank, setBank] = useState({ total_credits: 0, avg_monthly_balance: 0, emi_bounce_count: 0, avg_utilisation_pct: 65, credit_debit_ratio: 1.05 })
  const [field, setField] = useState({ factory_condition: 'good', management_quality: 'good', inventory_observation: '', workforce_observation: '', additional_notes: '' })

  const updateCompany = (k, v) => setCompany(p => ({ ...p, [k]: v }))
  const updateFinancials = (k, v) => setFinancials(p => ({ ...p, [k]: v }))
  const updateGst = (k, v) => setGst(p => ({ ...p, [k]: v }))
  const updateBank = (k, v) => setBank(p => ({ ...p, [k]: v }))
  const updateField = (k, v) => setField(p => ({ ...p, [k]: v }))

  const handleFileUpload = useCallback((e) => {
    setFiles(prev => [...prev, ...Array.from(e.target.files)])
  }, [])

  const removeFile = (idx) => setFiles(prev => prev.filter((_, i) => i !== idx))

  const uploadDocuments = async () => {
    for (const file of files) {
      const fd = new FormData()
      fd.append('file', file)
      fd.append('company_name', company.company_name)
      fd.append('doc_type', 'other')
      try { await fetch(`${API}/ingest/document`, { method: 'POST', body: fd }) } catch (e) { console.error(e) }
    }
  }

  // Classify uploaded documents
  const classifyDocuments = async () => {
    if (files.length === 0) return
    setClassifying(true)
    try {
      const fd = new FormData()
      files.forEach(f => fd.append('files', f))
      const res = await fetch(`${API}/classify`, { method: 'POST', body: fd })
      if (res.ok) {
        const data = await res.json()
        setClassifications(data.data.classifications)
        setDocTypes(data.data.doc_types)
      }
    } catch (e) { console.error(e) }
    finally { setClassifying(false) }
  }

  const updateClassification = (idx, newType) => {
    setClassifications(prev => prev.map((c, i) =>
      i === idx ? { ...c, confirmed_type: newType, confirmed: true } : c
    ))
  }

  // Load schemas
  const loadSchemas = async () => {
    try {
      const res = await fetch(`${API}/schema/defaults`)
      if (res.ok) {
        const data = await res.json()
        setSchemas(data.data)
        const keys = Object.keys(data.data)
        if (keys.length > 0 && !activeSchemaType) setActiveSchemaType(keys[0])
      }
    } catch (e) { console.error(e) }
  }

  const addSchemaField = (docType) => {
    setSchemas(prev => ({
      ...prev,
      [docType]: {
        ...prev[docType],
        fields: [...(prev[docType]?.fields || []), { name: '', label: '', type: 'text', required: false }]
      }
    }))
  }

  const removeSchemaField = (docType, idx) => {
    setSchemas(prev => ({
      ...prev,
      [docType]: {
        ...prev[docType],
        fields: prev[docType].fields.filter((_, i) => i !== idx)
      }
    }))
  }

  const updateSchemaField = (docType, idx, key, value) => {
    setSchemas(prev => ({
      ...prev,
      [docType]: {
        ...prev[docType],
        fields: prev[docType].fields.map((f, i) =>
          i === idx ? { ...f, [key]: value } : f
        )
      }
    }))
  }

  // Trigger classification when entering Document Review step
  useEffect(() => {
    if (step === 2 && files.length > 0 && classifications.length === 0) {
      classifyDocuments()
    }
  }, [step])

  // Load schemas when entering Schema Config step
  useEffect(() => {
    if (step === 3 && Object.keys(schemas).length === 0) {
      loadSchemas()
    }
  }, [step])

  // Download PDF report
  const downloadReport = async () => {
    setDownloading(true)
    try {
      const revHistory = financials.revenue_history.split(',').map(Number).filter(n => !isNaN(n))
      const body = {
        company_info: { ...company, promoter_names: company.promoter_names ? company.promoter_names.split(',').map(s => s.trim()) : [], years_in_business: Number(company.years_in_business), promoter_experience_yrs: Number(company.promoter_experience_yrs), cibil_score: Number(company.cibil_score), security_coverage: Number(company.security_coverage), macro_risk_score: Number(company.macro_risk_score) },
        financial_data: { ...Object.fromEntries(Object.entries(financials).filter(([k]) => k !== 'revenue_history').map(([k, v]) => [k, Number(v)])), revenue_history: revHistory },
      }
      const res = await fetch(`${API}/report/download`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (res.ok) {
        const blob = await res.blob()
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `CAM_${company.company_name.replace(/\s+/g, '_')}.pdf`
        document.body.appendChild(a)
        a.click()
        a.remove()
        window.URL.revokeObjectURL(url)
      }
    } catch (e) { console.error(e) }
    finally { setDownloading(false) }
  }

  const runAnalysis = async () => {
    setLoading(true); setError(''); setLoadingPhase(0)
    const interval = setInterval(() => setLoadingPhase(p => Math.min(p + 1, LOADING_PHASES.length - 1)), 2000)
    try {
      if (files.length > 0) await uploadDocuments()
      const revHistory = financials.revenue_history.split(',').map(Number).filter(n => !isNaN(n))
      const body = {
        company_info: { ...company, promoter_names: company.promoter_names ? company.promoter_names.split(',').map(s => s.trim()) : [], years_in_business: Number(company.years_in_business), promoter_experience_yrs: Number(company.promoter_experience_yrs), cibil_score: Number(company.cibil_score), security_coverage: Number(company.security_coverage), macro_risk_score: Number(company.macro_risk_score) },
        financial_data: { ...Object.fromEntries(Object.entries(financials).filter(([k]) => k !== 'revenue_history').map(([k, v]) => [k, Number(v)])), revenue_history: revHistory },
        gst_data: gst.gst_turnover > 0 ? { company_name: company.company_name, period: 'FY2023-24', ...Object.fromEntries(Object.entries(gst).map(([k, v]) => [k, Number(v)])) } : null,
        bank_data: bank.total_credits > 0 ? { ...Object.fromEntries(Object.entries(bank).map(([k, v]) => [k, Number(v)])) } : null,
        field_insights: field,
      }
      const res = await fetch(`${API}/decision`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
      if (!res.ok) { const err = await res.json(); throw new Error(err.detail || 'Analysis failed') }
      const data = await res.json()
      setResult(data.data)
      setStep(8)
    } catch (e) { setError(e.message) }
    finally { clearInterval(interval); setLoading(false) }
  }

  const canProceed = step === 0 ? company.company_name.length > 0 : true

  // ─── Page Renderers ───
  const renderCompany = () => (
    <div className="animate-in">
      <div className="page-header">
        <div className="page-title">Company Profile</div>
        <div className="page-subtitle">Enter borrower entity information for credit assessment</div>
      </div>
      <div className="card">
        <div className="card-header"><div className="card-title">Entity Information</div></div>
        <div className="form-grid">
          <Input label="Company Name *" value={company.company_name} onChange={v => updateCompany('company_name', v)} opts={{ placeholder: 'ABC Engineering Pvt Ltd' }} />
          <Input label="Industry / Sector" value={company.industry} onChange={v => updateCompany('industry', v)} opts={{ placeholder: 'Manufacturing' }} />
          <Input label="CIN" value={company.cin} onChange={v => updateCompany('cin', v)} opts={{ placeholder: 'U72200KA2015PTC...' }} />
          <Input label="Years in Business" value={company.years_in_business} onChange={v => updateCompany('years_in_business', v)} type="number" />
        </div>
      </div>
      <div className="card">
        <div className="card-header"><div className="card-title">Promoter & Credit Profile</div></div>
        <div className="form-grid">
          <Input label="Promoter / Director Names" value={company.promoter_names} onChange={v => updateCompany('promoter_names', v)} opts={{ placeholder: 'Rajesh Kumar, Suresh Patel' }} />
          <Input label="Promoter Experience (Years)" value={company.promoter_experience_yrs} onChange={v => updateCompany('promoter_experience_yrs', v)} type="number" />
          <Input label="CIBIL Commercial Score" value={company.cibil_score} onChange={v => updateCompany('cibil_score', v)} type="number" />
          <Input label="Collateral Type" value={company.collateral_type} onChange={v => updateCompany('collateral_type', v)} opts={{ select: true, options: ['immovable', 'movable', 'unsecured'] }} />
          <Input label="Security Coverage (x)" value={company.security_coverage} onChange={v => updateCompany('security_coverage', v)} type="number" />
          <Input label="Sector Outlook" value={company.sector_outlook} onChange={v => updateCompany('sector_outlook', v)} opts={{ select: true, options: ['positive', 'stable', 'negative'] }} />
          <Input label="Macro Risk Score (1–10)" value={company.macro_risk_score} onChange={v => updateCompany('macro_risk_score', v)} type="number" />
        </div>
      </div>
    </div>
  )

  const renderDocuments = () => (
    <div className="animate-in">
      <div className="page-header">
        <div className="page-title">Document Upload</div>
        <div className="page-subtitle">Upload borrower documents for automated extraction and RAG analysis</div>
      </div>
      <div className="card">
        <div className="card-header"><div className="card-title">Upload Documents</div></div>
        <label className="upload-zone">
          <input type="file" accept=".pdf" multiple onChange={handleFileUpload} style={{ display: 'none' }} />
          <div className="upload-icon">↑</div>
          <div className="upload-text">Click to select PDF files or drag & drop</div>
          <div className="upload-hint">Accepted: Annual Reports, Financial Statements, Bank Statements, GST Returns, Legal Documents, MCA Filings</div>
        </label>
        {files.length > 0 && (
          <div className="file-list">
            {files.map((f, i) => (
              <div className="file-item" key={i}>
                <span>PDF — {f.name} ({(f.size / 1024).toFixed(0)} KB)</span>
                <button className="file-remove" onClick={() => removeFile(i)}>×</button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )

  // ─── NEW: Document Classification & HITL Review ───
  const renderDocReview = () => (
    <div className="animate-in">
      <div className="page-header">
        <div className="page-title">Document Review & Classification</div>
        <div className="page-subtitle">AI has auto-classified your documents. Review, approve, or override each classification below.</div>
      </div>
      {files.length === 0 ? (
        <div className="card">
          <div style={{ padding: '24px', textAlign: 'center', color: '#a0aec0' }}>
            <div style={{ fontSize: '1.2rem', marginBottom: 6 }}>No documents uploaded</div>
            <div style={{ fontSize: '0.82rem' }}>Go back to the Document Upload step to add files.</div>
          </div>
        </div>
      ) : classifying ? (
        <div className="card">
          <div className="loading-overlay" style={{ padding: '32px' }}>
            <div className="spinner" />
            <div className="loading-text">Classifying documents with AI...</div>
          </div>
        </div>
      ) : (
        <div className="card">
          <div className="card-header">
            <div className="card-title">Classification Results</div>
            <div className="card-badge" style={{ background: '#ebf8ff', color: '#2b6cb0' }}>Human-in-the-Loop</div>
          </div>
          <div className="classification-list">
            {classifications.map((c, i) => {
              const confPct = Math.round((c.confidence || 0) * 100)
              const confColor = confPct >= 70 ? '#38a169' : confPct >= 40 ? '#d69e2e' : '#e53e3e'
              return (
                <div className="classification-item" key={i}>
                  <div className="classification-file">
                    <span className="classification-file-icon">📄</span>
                    <span>{c.filename}</span>
                  </div>
                  <div className="classification-details">
                    <div className="classification-row">
                      <span className="classification-label">AI Suggestion:</span>
                      <span className="classification-badge" style={{ background: confColor + '18', color: confColor, borderColor: confColor }}>
                        {c.suggested_label} ({confPct}%)
                      </span>
                    </div>
                    <div className="classification-row">
                      <span className="classification-label">Confirmed Type:</span>
                      <select
                        className="classification-select"
                        value={c.confirmed_type || c.suggested_type}
                        onChange={e => updateClassification(i, e.target.value)}
                      >
                        {(docTypes.length > 0 ? docTypes : [{ value: c.suggested_type, label: c.suggested_label }]).map(dt => (
                          <option key={dt.value} value={dt.value}>{dt.label}</option>
                        ))}
                      </select>
                    </div>
                  </div>
                  <div className="classification-status">
                    {c.confirmed ? (
                      <span style={{ color: '#38a169', fontWeight: 600, fontSize: '0.78rem' }}>✓ Confirmed</span>
                    ) : (
                      <span style={{ color: '#d69e2e', fontSize: '0.78rem' }}>Pending review</span>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
          {classifications.length > 0 && (
            <div className="btn-group" style={{ marginTop: 16 }}>
              <button className="btn btn-primary btn-sm" onClick={() => {
                setClassifications(prev => prev.map(c => ({ ...c, confirmed: true, confirmed_type: c.confirmed_type || c.suggested_type })))
              }}>
                Approve All Classifications
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  )

  // ─── NEW: Dynamic Schema Configuration ───
  const renderSchemaConfig = () => (
    <div className="animate-in">
      <div className="page-header">
        <div className="page-title">Schema Configuration</div>
        <div className="page-subtitle">Define the data fields to extract from each document type. Add, remove, or rename fields as needed.</div>
      </div>
      <div className="card">
        <div className="card-header">
          <div className="card-title">Extraction Schemas</div>
          <div className="card-badge" style={{ background: '#f0fff4', color: '#38a169' }}>Dynamic Schema</div>
        </div>
        <div className="schema-tabs">
          {Object.entries(schemas).map(([key, schema]) => (
            <button
              key={key}
              className={`schema-tab ${activeSchemaType === key ? 'active' : ''}`}
              onClick={() => setActiveSchemaType(key)}
            >
              {schema.label || key}
            </button>
          ))}
        </div>
        {activeSchemaType && schemas[activeSchemaType] && (
          <div className="schema-editor">
            <div className="schema-fields">
              {(schemas[activeSchemaType]?.fields || []).map((f, i) => (
                <div className="schema-field-row" key={i}>
                  <input
                    className="schema-input"
                    value={f.label}
                    onChange={e => updateSchemaField(activeSchemaType, i, 'label', e.target.value)}
                    placeholder="Field Label"
                  />
                  <select
                    className="schema-input schema-type-select"
                    value={f.type}
                    onChange={e => updateSchemaField(activeSchemaType, i, 'type', e.target.value)}
                  >
                    <option value="text">Text</option>
                    <option value="number">Number</option>
                    <option value="date">Date</option>
                  </select>
                  <label className="schema-required">
                    <input
                      type="checkbox"
                      checked={f.required}
                      onChange={e => updateSchemaField(activeSchemaType, i, 'required', e.target.checked)}
                    />
                    Required
                  </label>
                  <button className="schema-remove-btn" onClick={() => removeSchemaField(activeSchemaType, i)}>×</button>
                </div>
              ))}
            </div>
            <button className="btn btn-secondary btn-sm" style={{ marginTop: 12 }} onClick={() => addSchemaField(activeSchemaType)}>
              + Add Field
            </button>
          </div>
        )}
      </div>
    </div>
  )

  const renderGstBank = () => (
    <div className="animate-in">
      <div className="page-header">
        <div className="page-title">GST & Banking Data</div>
        <div className="page-subtitle">Structured data for GST cross-validation and bank statement analysis</div>
      </div>
      <div className="card">
        <div className="card-header"><div className="card-title">GST Return Data</div><div className="card-badge" style={{ background: '#ebf8ff', color: '#2b6cb0' }}>GSTR-2A vs GSTR-3B</div></div>
        <div className="form-grid">
          <Input label="GSTR-2A ITC (₹ Lakhs)" value={gst.gstr_2a_itc} onChange={v => updateGst('gstr_2a_itc', v)} type="number" />
          <Input label="GSTR-3B ITC (₹ Lakhs)" value={gst.gstr_3b_itc} onChange={v => updateGst('gstr_3b_itc', v)} type="number" />
          <Input label="GST Turnover (₹ Lakhs)" value={gst.gst_turnover} onChange={v => updateGst('gst_turnover', v)} type="number" />
        </div>
      </div>
      <div className="card">
        <div className="card-header"><div className="card-title">Bank Statement Indicators</div></div>
        <div className="form-grid">
          <Input label="Total Credits (₹ Lakhs)" value={bank.total_credits} onChange={v => updateBank('total_credits', v)} type="number" />
          <Input label="Avg Monthly Balance (₹ Lakhs)" value={bank.avg_monthly_balance} onChange={v => updateBank('avg_monthly_balance', v)} type="number" />
          <Input label="EMI/ECS Bounce Count (12M)" value={bank.emi_bounce_count} onChange={v => updateBank('emi_bounce_count', v)} type="number" />
          <Input label="Avg CC Utilisation %" value={bank.avg_utilisation_pct} onChange={v => updateBank('avg_utilisation_pct', v)} type="number" />
          <Input label="Credit / Debit Ratio" value={bank.credit_debit_ratio} onChange={v => updateBank('credit_debit_ratio', v)} type="number" />
        </div>
      </div>
    </div>
  )

  const renderFieldInsights = () => (
    <div className="animate-in">
      <div className="page-header">
        <div className="page-title">Primary / Field Insights</div>
        <div className="page-subtitle">Qualitative observations from site visit and management assessment</div>
      </div>
      <div className="card">
        <div className="card-header"><div className="card-title">Assessment Observations</div></div>
        <div className="form-grid">
          <Input label="Factory / Site Condition" value={field.factory_condition} onChange={v => updateField('factory_condition', v)} opts={{ select: true, options: ['excellent', 'good', 'fair', 'poor'] }} />
          <Input label="Management Quality" value={field.management_quality} onChange={v => updateField('management_quality', v)} opts={{ select: true, options: ['excellent', 'good', 'fair', 'poor'] }} />
        </div>
        <div style={{ marginTop: 14 }}>
          <div className="form-grid">
            <Input label="Inventory Observation" value={field.inventory_observation} onChange={v => updateField('inventory_observation', v)} opts={{ textarea: true, placeholder: 'Inventory levels, quality, storage conditions...' }} />
            <Input label="Workforce Observation" value={field.workforce_observation} onChange={v => updateField('workforce_observation', v)} opts={{ textarea: true, placeholder: 'Workforce size, skill level, operational activity...' }} />
          </div>
          <div style={{ marginTop: 14 }}>
            <Input label="Additional Notes" value={field.additional_notes} onChange={v => updateField('additional_notes', v)} opts={{ textarea: true, placeholder: 'Any other relevant field observations...' }} />
          </div>
        </div>
      </div>
    </div>
  )

  const renderFinancials = () => (
    <div className="animate-in">
      <div className="page-header">
        <div className="page-title">Financial Statements</div>
        <div className="page-subtitle">Key financial data from latest audited accounts (₹ Crores)</div>
      </div>
      <div className="card">
        <div className="card-header"><div className="card-title">Profit & Loss</div></div>
        <div className="form-grid">
          <Input label="Revenue (₹ Cr)" value={financials.revenue_cr} onChange={v => updateFinancials('revenue_cr', v)} type="number" />
          <Input label="EBITDA (₹ Cr)" value={financials.ebitda_cr} onChange={v => updateFinancials('ebitda_cr', v)} type="number" />
          <Input label="PAT (₹ Cr)" value={financials.pat_cr} onChange={v => updateFinancials('pat_cr', v)} type="number" />
          <Input label="Interest Expense (₹ Cr)" value={financials.interest_expense_cr} onChange={v => updateFinancials('interest_expense_cr', v)} type="number" />
          <Input label="Depreciation (₹ Cr)" value={financials.depreciation_cr} onChange={v => updateFinancials('depreciation_cr', v)} type="number" />
        </div>
      </div>
      <div className="card">
        <div className="card-header"><div className="card-title">Balance Sheet</div></div>
        <div className="form-grid">
          <Input label="Total Debt (₹ Cr)" value={financials.total_debt_cr} onChange={v => updateFinancials('total_debt_cr', v)} type="number" />
          <Input label="Net Worth (₹ Cr)" value={financials.net_worth_cr} onChange={v => updateFinancials('net_worth_cr', v)} type="number" />
          <Input label="Current Assets (₹ Cr)" value={financials.current_assets_cr} onChange={v => updateFinancials('current_assets_cr', v)} type="number" />
          <Input label="Current Liabilities (₹ Cr)" value={financials.current_liabilities_cr} onChange={v => updateFinancials('current_liabilities_cr', v)} type="number" />
          <Input label="Inventory (₹ Cr)" value={financials.inventory_cr} onChange={v => updateFinancials('inventory_cr', v)} type="number" />
          <Input label="Fixed Assets (₹ Cr)" value={financials.fixed_assets_cr} onChange={v => updateFinancials('fixed_assets_cr', v)} type="number" />
        </div>
      </div>
      <div className="card">
        <div className="card-header"><div className="card-title">Debt Service & Capital</div></div>
        <div className="form-grid">
          <Input label="Annual Debt Service (₹ Cr)" value={financials.annual_debt_service_cr} onChange={v => updateFinancials('annual_debt_service_cr', v)} type="number" />
          <Input label="Promoter Contribution %" value={financials.promoter_contribution_pct} onChange={v => updateFinancials('promoter_contribution_pct', v)} type="number" />
          <Input label="Revenue History (comma-separated)" value={financials.revenue_history} onChange={v => updateFinancials('revenue_history', v)} opts={{ placeholder: '7,8,9,10' }} />
        </div>
      </div>
    </div>
  )

  const renderAnalyze = () => (
    <div className="animate-in">
      <div className="page-header">
        <div className="page-title">Credit Analysis Pipeline</div>
        <div className="page-subtitle">Execute multi-agent analysis, ML scoring, and credit committee synthesis</div>
      </div>
      <div className="card">
        {loading ? (
          <div className="loading-overlay">
            <div className="spinner" />
            <div className="loading-text">Executing Credit Analysis Pipeline</div>
            <div className="loading-steps">
              {LOADING_PHASES.map((phase, i) => (
                <div key={i} className={`loading-step ${i < loadingPhase ? 'done' : i === loadingPhase ? 'active' : ''}`}>
                  {i < loadingPhase ? '✓' : i === loadingPhase ? '›' : '·'} {phase}
                </div>
              ))}
            </div>
          </div>
        ) : (
          <>
            <div className="card-header"><div className="card-title">Analysis Configuration</div></div>
            {error && <div style={{ color: '#e53e3e', padding: '10px 14px', background: '#fff5f5', border: '1px solid #fed7d7', borderRadius: '4px', marginBottom: '14px', fontSize: '0.82rem' }}>Error: {error}</div>}
            <div className="pipeline-info">
              <div style={{ fontSize: '0.78rem', fontWeight: 600, color: '#2d3748', marginBottom: '8px' }}>Pipeline Stages</div>
              {['Financial ratio computation (DSCR, D/E, ICR, CR)', 'Secondary research — news, MCA, litigation (Tavily)', 'Fraud graph construction & network analysis (NetworkX)', 'Cashflow forecasting (Prophet time-series)', 'Risk signal aggregation across all data sources', 'ML ensemble scoring — XGBoost + LightGBM + RF', 'SHAP feature attribution & explainability', 'AI credit committee synthesis (GPT-4o)', 'SWOT analysis generation', 'GenAI executive narrative', 'Credit Appraisal Memo generation'].map((s, i) => (
                <div key={i} className="pipeline-item">{s}</div>
              ))}
            </div>
            <div className="btn-group">
              <button className="btn btn-primary" onClick={runAnalysis} disabled={!company.company_name}>
                Execute Analysis
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  )

  const renderDecision = () => {
    if (!result) return null
    const { decision: d, agent_analysis: aa, cam, swot_analysis: swot, genai_narrative: genaiNarr } = result
    const fiveCs = aa?.committee_synthesis?.five_cs || {}
    const decisionClass = d?.decision?.includes('APPROVE') && !d?.decision?.includes('CONDITIONS') ? 'approve' : d?.decision?.includes('CONDITIONS') ? 'conditions' : 'decline'

    return (
      <div className="animate-in">
        <div className="page-header">
          <div className="page-title">Credit Decision — {company.company_name}</div>
          <div className="page-subtitle">ML-driven credit assessment with explainable AI</div>
        </div>

        <div className={`decision-banner ${decisionClass}`}>
          <div className="decision-main">
            <div className="decision-label">Recommendation</div>
            <div className="decision-value">{d?.decision || 'N/A'}</div>
          </div>
          <div className="decision-meta">
            <div className="decision-grade">Grade {d?.credit_grade} · Score {d?.credit_score}/900</div>
            <div className="decision-grade">Default Probability: {d?.default_probability}%</div>
          </div>
        </div>

        <div className="metrics-grid">
          {[
            ['Loan Amount', `₹${d?.final_loan_amount_cr} Cr`, ''],
            ['Interest Rate', `${d?.interest_rate_pct}%`, 'per annum'],
            ['Default Probability', `${d?.default_probability}%`, 'ML Ensemble'],
            ['Haircut Applied', `${d?.haircut_pct}%`, 'Security adj.'],
            ['Credit Score', `${d?.credit_score}`, '/ 900'],
            ['Five Cs Total', `${fiveCs.total?.score || 0}`, '/ 100'],
          ].map(([label, value, unit], i) => (
            <div className="metric-card" key={i}>
              <div className="metric-label">{label}</div>
              <div className="metric-value">{value}</div>
              {unit && <div className="metric-unit">{unit}</div>}
            </div>
          ))}
        </div>

        <div className="tab-group">
          {[['summary', 'Summary'], ['shap', 'Model Explanation'], ['fivecs', 'Five Cs Scorecard'], ['risks', 'Risk Assessment'], ['fraud', 'Fraud Analysis'], ['swot', 'SWOT Analysis'], ['genai', 'GenAI Insights'], ['memo', 'Credit Memo']].map(([key, label]) => (
            <button key={key} className={`tab-btn ${activeTab === key ? 'active' : ''}`} onClick={() => setActiveTab(key)}>{label}</button>
          ))}
        </div>

        {activeTab === 'summary' && (
          <div className="card">
            <div className="card-header"><div className="card-title">Pricing Breakdown</div></div>
            {d?.pricing_breakdown && (
              <div className="pricing-breakdown">
                <div className="pricing-row"><span>Grade Band</span><span>{d.pricing_breakdown.grade_band}</span></div>
                <div className="pricing-row"><span>Base Rate</span><span>{d.pricing_breakdown.base_rate}%</span></div>
                {(d.pricing_breakdown.adjustments || []).map((a, i) => (
                  <div className="pricing-row" key={i}><span>{a.factor}</span><span>{a.adjustment > 0 ? '+' : ''}{a.adjustment}%</span></div>
                ))}
                <div className="pricing-row total"><span>Final Rate</span><span>{d?.interest_rate_pct}%</span></div>
              </div>
            )}
            <div className="section-divider" />
            <div className="section-title">Suggested Covenants</div>
            <div className="covenant-list">
              {(d?.covenants || []).map((c, i) => <div className="covenant-item" key={i}><span style={{ color: '#718096' }}>{i + 1}.</span><span>{c}</span></div>)}
            </div>
          </div>
        )}

        {activeTab === 'shap' && (
          <div className="card">
            <div className="card-header"><div className="card-title">SHAP Feature Attribution</div><div className="card-badge" style={{ background: '#ebf8ff', color: '#2b6cb0' }}>XGBoost + TreeExplainer</div></div>
            <div style={{ fontSize: '0.78rem', color: '#718096', marginBottom: 14 }}>Impact of each feature on the predicted default probability. Red bars indicate risk-increasing factors; green bars indicate protective factors.</div>
            <div className="shap-list">
              {(d?.shap_explanation || []).map((s, i) => {
                const maxImpact = Math.max(...(d?.shap_explanation || []).map(x => Math.abs(x.impact)), 0.01)
                const widthPct = Math.min(100, (Math.abs(s.impact) / maxImpact) * 100)
                return (
                  <div className="shap-item" key={i}>
                    <span className="shap-feature">{s.feature.replace(/_/g, ' ')}</span>
                    <div className="shap-bar-container">
                      <div className={`shap-bar ${s.impact > 0 ? 'positive' : 'negative'}`} style={{ width: `${widthPct}%` }} />
                    </div>
                    <span className="shap-value" style={{ color: s.impact > 0 ? '#e53e3e' : '#38a169' }}>
                      {s.impact > 0 ? '+' : ''}{s.impact.toFixed(4)}
                    </span>
                  </div>
                )
              })}
            </div>
          </div>
        )}

        {activeTab === 'fivecs' && (
          <div className="card">
            <div className="card-header"><div className="card-title">Five Cs of Credit — Scorecard</div></div>
            <div className="five-cs-grid">
              {['character', 'capacity', 'capital', 'collateral', 'conditions'].map(c => {
                const data = fiveCs[c] || { score: 0, max: 20 }
                const pct = (data.score / data.max) * 100
                const color = pct >= 70 ? '#38a169' : pct >= 50 ? '#d69e2e' : '#e53e3e'
                return (
                  <div className="five-c-item" key={c}>
                    <div className="five-c-name">{c}</div>
                    <div className="five-c-score" style={{ color }}>{data.score}</div>
                    <div className="five-c-max">/ {data.max}</div>
                    <div className="five-c-bar"><div className="five-c-fill" style={{ width: `${pct}%`, background: color }} /></div>
                  </div>
                )
              })}
            </div>
          </div>
        )}

        {activeTab === 'risks' && (
          <div className="card">
            <div className="card-header"><div className="card-title">Risk Factor Analysis</div><div className="card-badge" style={{ background: aa?.risk_assessment?.risk_level === 'HIGH' || aa?.risk_assessment?.risk_level === 'CRITICAL' ? '#fff5f5' : '#fffff0', color: aa?.risk_assessment?.risk_level === 'HIGH' || aa?.risk_assessment?.risk_level === 'CRITICAL' ? '#e53e3e' : '#d69e2e' }}>{aa?.risk_assessment?.risk_level || 'N/A'}</div></div>
            <div className="risk-list">
              {(aa?.risk_assessment?.risk_factors || []).map((r, i) => (
                <div className={`risk-item ${r.severity}`} key={i}>
                  <span className={`risk-severity ${r.severity}`}>{r.severity}</span>
                  <div>
                    <div style={{ fontWeight: 600, fontSize: '0.82rem', color: '#2d3748' }}>{r.factor}</div>
                    <div className="risk-detail">{r.detail}</div>
                  </div>
                </div>
              ))}
              {(aa?.risk_assessment?.risk_factors || []).length === 0 && (
                <div style={{ color: '#a0aec0', fontSize: '0.82rem', padding: '8px' }}>No significant risk factors identified.</div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'fraud' && (
          <div className="card">
            <div className="card-header"><div className="card-title">Fraud Network Analysis</div><div className="card-badge" style={{ background: '#ebf8ff', color: '#2b6cb0' }}>NetworkX Graph Analytics</div></div>
            {aa?.fraud_analysis && (
              <>
                <div className="metrics-grid" style={{ marginBottom: 16 }}>
                  <div className="metric-card"><div className="metric-label">Graph Risk Level</div><div className="metric-value">{aa.fraud_analysis.risk_level}</div></div>
                  <div className="metric-card"><div className="metric-label">Risk Score</div><div className="metric-value">{aa.fraud_analysis.overall_risk_score}<span className="metric-unit"> / 10</span></div></div>
                </div>
                {(aa.fraud_analysis.recommendations || []).length > 0 && (
                  <>
                    <div className="section-title">Recommendations</div>
                    <div className="covenant-list">
                      {aa.fraud_analysis.recommendations.map((r, i) => <div className="covenant-item" key={i}><span style={{ color: '#718096' }}>→</span><span>{r}</span></div>)}
                    </div>
                  </>
                )}
              </>
            )}
          </div>
        )}

        {/* ─── NEW: SWOT Analysis Tab ─── */}
        {activeTab === 'swot' && (
          <div className="card">
            <div className="card-header">
              <div className="card-title">SWOT Analysis</div>
              <div className="card-badge" style={{ background: '#f0fff4', color: '#38a169' }}>Strategic Assessment</div>
            </div>
            {swot ? (
              <>
                <div className="swot-grid">
                  <div className="swot-quadrant swot-strengths">
                    <div className="swot-quadrant-title">💪 Strengths</div>
                    {(swot.strengths || []).map((s, i) => (
                      <div className="swot-item" key={i}>
                        <div className="swot-item-title">{s.title}</div>
                        <div className="swot-item-detail">{s.detail}</div>
                      </div>
                    ))}
                  </div>
                  <div className="swot-quadrant swot-weaknesses">
                    <div className="swot-quadrant-title">⚠️ Weaknesses</div>
                    {(swot.weaknesses || []).map((s, i) => (
                      <div className="swot-item" key={i}>
                        <div className="swot-item-title">{s.title}</div>
                        <div className="swot-item-detail">{s.detail}</div>
                      </div>
                    ))}
                  </div>
                  <div className="swot-quadrant swot-opportunities">
                    <div className="swot-quadrant-title">🚀 Opportunities</div>
                    {(swot.opportunities || []).map((s, i) => (
                      <div className="swot-item" key={i}>
                        <div className="swot-item-title">{s.title}</div>
                        <div className="swot-item-detail">{s.detail}</div>
                      </div>
                    ))}
                  </div>
                  <div className="swot-quadrant swot-threats">
                    <div className="swot-quadrant-title">🔴 Threats</div>
                    {(swot.threats || []).map((s, i) => (
                      <div className="swot-item" key={i}>
                        <div className="swot-item-title">{s.title}</div>
                        <div className="swot-item-detail">{s.detail}</div>
                      </div>
                    ))}
                  </div>
                </div>
                {swot.summary && (
                  <div style={{ marginTop: 16, padding: '12px 16px', background: '#f7f8fa', borderRadius: '6px', fontSize: '0.82rem', color: '#4a5568', lineHeight: 1.6 }}>
                    <strong>Summary: </strong>{swot.summary}
                  </div>
                )}
              </>
            ) : (
              <div style={{ color: '#a0aec0', fontSize: '0.82rem', padding: 8 }}>SWOT analysis not available.</div>
            )}
          </div>
        )}

        {/* ─── NEW: GenAI Insights Tab ─── */}
        {activeTab === 'genai' && (
          <div className="card">
            <div className="card-header">
              <div className="card-title">GenAI Executive Summary</div>
              <div className="card-badge" style={{ background: '#f0e6ff', color: '#6b46c1' }}>
                {genaiNarr?.generated_by || 'AI Generated'}
              </div>
            </div>
            {genaiNarr?.narrative ? (
              <div className="genai-narrative">
                {genaiNarr.narrative}
              </div>
            ) : (
              <div style={{ color: '#a0aec0', fontSize: '0.82rem', padding: 8 }}>GenAI narrative not available.</div>
            )}
            {genaiNarr?.generated_at && (
              <div style={{ marginTop: 12, fontSize: '0.72rem', color: '#a0aec0' }}>
                Generated: {new Date(genaiNarr.generated_at).toLocaleString()}
              </div>
            )}
          </div>
        )}

        {activeTab === 'memo' && (
          <div className="card">
            <div className="card-header">
              <div className="card-title">Credit Appraisal Memo (CAM)</div>
              <button
                className="btn btn-primary btn-sm"
                onClick={downloadReport}
                disabled={downloading}
                style={{ marginLeft: 'auto' }}
              >
                {downloading ? 'Generating PDF...' : '⬇ Download PDF Report'}
              </button>
            </div>
            <div className="cam-memo">{cam?.memo_text || 'No memo generated.'}</div>
          </div>
        )}
      </div>
    )
  }

  const stepRenderers = [renderCompany, renderDocuments, renderDocReview, renderSchemaConfig, renderGstBank, renderFieldInsights, renderFinancials, renderAnalyze, renderDecision]

  // Derive sidebar sections
  let lastSection = ''

  return (
    <div className="app">
      {/* Sidebar */}
      <nav className="sidebar">
        <div className="sidebar-brand">
          <div className="sidebar-brand-name">Intelli-Credit</div>
          <div className="sidebar-brand-sub">Credit Decisioning Engine</div>
        </div>

        {STEPS.map((s, i) => {
          const showSection = s.section !== lastSection
          lastSection = s.section
          return (
            <div key={i}>
              {showSection && <div className="sidebar-section-label">{s.section}</div>}
              <div className="sidebar-nav">
                <button
                  className={`sidebar-item ${i === step ? 'active' : ''} ${i < step ? 'completed' : ''}`}
                  onClick={() => { if (i <= step || i < step) setStep(i) }}
                >
                  <span className="sidebar-step-num">{i < step ? '✓' : i + 1}</span>
                  <span>{s.label}</span>
                </button>
              </div>
            </div>
          )
        })}

        <div className="sidebar-footer">
          <div className="sidebar-status">
            <span className="status-indicator" />
            <span>ML Models Active</span>
          </div>
        </div>
      </nav>

      {/* Main */}
      <div className="main-content">
        <div className="top-bar">
          <div>
            <div className="top-bar-title">{STEPS[step].label}</div>
            <div className="top-bar-breadcrumb">Credit Analysis / {STEPS[step].section} / {STEPS[step].label}</div>
          </div>
          <div className="top-bar-actions">
            {step < 7 && step > 0 && <button className="btn btn-secondary btn-sm" onClick={() => setStep(s => s - 1)}>← Previous</button>}
            {step < 7 && <button className="btn btn-primary btn-sm" onClick={() => setStep(s => s + 1)} disabled={!canProceed}>Next Step →</button>}
          </div>
        </div>

        <div className="page-content">
          {stepRenderers[step]?.()}
        </div>
      </div>
    </div>
  )
}
