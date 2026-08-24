"""Dashboard styling and theme constants."""

# Default colors used in Python/Charts (Plotly)
# Plotly supports template='plotly_white' or 'plotly_dark'
COLORS = {
    'primary': '#1a365d',     # Navy
    'secondary': '#0d9488',   # Teal
    'accent': '#d97706',      # Amber
    'danger': '#dc2626',      # Muted Red
    'text': '#0f172a',        # Slate 900
    'muted': '#64748b',       # Slate 500
    'bg': '#ffffff',          # White
    'surface': '#f8fafc',     # Slate 50
    'border': '#e2e8f0'       # Slate 200
}

# Add a utility for adaptive CSS
def get_custom_css():
    return """
    <style>
        /* Base typography and spacing */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
            max-width: 1200px !important;
        }
        
        /* Base Variables (Light Mode Default) */
        :root {
            --giip-bg: #ffffff;
            --giip-surface: #f8fafc;
            --giip-text: #0f172a;
            --giip-text-muted: #64748b;
            --giip-border: #e2e8f0;
            --giip-primary: #1a365d;
            --giip-signal-bg: #f8fafc;
            --giip-signal-border: #1a365d;
        }

        /* Dark Mode Override */
        @media (prefers-color-scheme: dark) {
            :root {
                --giip-bg: #0e1117;
                --giip-surface: #262730;
                --giip-text: #fafafa;
                --giip-text-muted: #a1a1aa;
                --giip-border: #3f3f46;
                --giip-primary: #38bdf8;
                --giip-signal-bg: #1e293b;
                --giip-signal-border: #38bdf8;
            }
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: var(--giip-text) !important;
            font-weight: 600 !important;
            letter-spacing: -0.02em !important;
            margin-bottom: 0.5rem !important;
        }

        h1 {
            font-size: 1.5rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
            border-bottom: 1px solid var(--giip-border);
            padding-bottom: 0.5rem;
            margin-bottom: 1.5rem !important;
        }

        h2 { font-size: 1.25rem !important; }
        h3 { font-size: 1.1rem !important; }
        
        p, div {
            color: var(--giip-text);
        }
        
        .subtext {
            color: var(--giip-text-muted) !important;
        }

        /* Remove default Streamlit styling elements */
        .st-emotion-cache-1wivap2 { display: none; }
        .stMetric { background-color: transparent !important; }
        
        /* Custom Decision Summary Cards */
        .decision-card {
            background-color: var(--giip-bg);
            border: 1px solid var(--giip-border);
            padding: 1.25rem;
            margin-bottom: 1rem;
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }
        .decision-label {
            font-size: 0.75rem;
            color: var(--giip-text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 600;
        }
        .decision-value {
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--giip-text);
        }
        .decision-subtext {
            font-size: 0.875rem;
            color: var(--giip-text-muted);
        }

        /* Key Signals */
        .signal-box {
            border-left: 3px solid var(--giip-signal-border);
            padding: 1rem;
            background-color: var(--giip-signal-bg);
            margin-bottom: 1rem;
        }
        .signal-title {
            font-size: 0.95rem;
            font-weight: 600;
            color: var(--giip-text);
            margin-bottom: 0.25rem;
        }
        .signal-value {
            font-size: 1.1rem;
            color: #0d9488;
            font-weight: 500;
        }

        /* Scenario Workspace */
        .scenario-box {
            border: 1px solid var(--giip-border);
            padding: 1.5rem;
            background-color: var(--giip-bg);
        }
        .scenario-header {
            font-size: 0.875rem;
            font-weight: 600;
            text-transform: uppercase;
            color: var(--giip-text-muted);
            margin-bottom: 1rem;
            letter-spacing: 0.05em;
            border-bottom: 1px solid var(--giip-border);
            padding-bottom: 0.5rem;
        }
        
        /* Tables */
        .dataframe {
            border: none !important;
            border-collapse: collapse !important;
            width: 100% !important;
        }
        .dataframe th {
            border: none !important;
            border-bottom: 1px solid var(--giip-border) !important;
            background-color: var(--giip-bg) !important;
            color: var(--giip-text-muted) !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            font-size: 0.75rem !important;
            padding: 0.75rem 0.5rem !important;
            text-align: right !important;
        }
        .dataframe th:first-child { text-align: left !important; }
        .dataframe td {
            border: none !important;
            border-bottom: 1px solid var(--giip-border) !important;
            padding: 0.75rem 0.5rem !important;
            color: var(--giip-text) !important;
            font-size: 0.875rem !important;
            text-align: right !important;
        }
        .dataframe td:first-child { text-align: left !important; }
        
        /* Sidebar styling */
        .css-1544g2n {
            padding-top: 2rem !important;
        }
        
        hr {
            margin: 1.5rem 0;
            border-color: var(--giip-border);
        }
    </style>
    """
