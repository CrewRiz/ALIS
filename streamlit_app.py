
"""
Streamlit interface for ALIS system.
"""

import streamlit as st
import asyncio
from typing import Dict, Any, List
import logging
from datetime import datetime, timezone
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import networkx as nx

from enhanced_learning_system import EnhancedLearningSystem
from settings import SETTINGS

# Configure page settings
st.set_page_config(
    page_title="ALIS - Advanced Learning System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize session state variables"""
    if 'system' not in st.session_state:
        st.session_state.system = EnhancedLearningSystem()
    if 'task_history' not in st.session_state:
        st.session_state.task_history = []
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "Task"
    if 'theme' not in st.session_state:
        st.session_state.theme = "light"

def render_sidebar():
    """Render sidebar with system information"""
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/your-repo/alis/main/assets/logo.png", 
                use_column_width=True)
        
        # System Stats
        st.header("System Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Complexity", 
                     f"{st.session_state.system.complexity:.2f}",
                     f"{st.session_state.system.complexity - st.session_state.system.last_complexity:.2f}")
        with col2:
            st.metric("Novelty",
                     f"{st.session_state.system.novelty:.2f}",
                     f"{st.session_state.system.novelty - st.session_state.system.last_novelty:.2f}")
        
        # Quantum State
        st.subheader("Quantum State")
        quantum_state = st.session_state.system.consciousness.quantum_state
        quantum_fig = go.Figure(data=[
            go.Scatterpolar(
                r=[quantum_state['wave_function'], 
                   quantum_state['temporal_coherence'],
                   len(quantum_state['superposition']) / 10,
                   len(quantum_state['entangled_thoughts']) / 10],
                theta=['Wave Function', 'Coherence', 'Superposition', 'Entanglement'],
                fill='toself'
            )
        ])
        quantum_fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(quantum_fig, use_container_width=True)
        
        # Settings
        st.subheader("Settings")
        theme = st.selectbox("Theme", ["light", "dark"], 
                           index=0 if st.session_state.theme == "light" else 1)
        if theme != st.session_state.theme:
            st.session_state.theme = theme
            st.experimental_rerun()

def render_task_interface():
    """Render main task interface"""
    st.header("Task Interface")
    
    # Task input
    task = st.text_area(
        "Enter your task or question:",
        help="Describe what you'd like ALIS to help you with. Be as specific as possible."
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        task_type = st.selectbox(
            "Task Type",
            ["General Query", "Web Interaction", "Pattern Analysis", "Learning Task"]
        )
    with col2:
        priority = st.slider("Priority", 1, 5, 3)
    
    if st.button("Process Task", type="primary"):
        if task:
            with st.spinner("Processing your request..."):
                try:
                    results = asyncio.run(
                        st.session_state.system.process_web_task(
                            task, 
                            task_type=task_type,
                            priority=priority
                        )
                    )
                    
                    # Add to history
                    st.session_state.task_history.append({
                        'task': task,
                        'type': task_type,
                        'results': results,
                        'timestamp': SETTINGS.get_current_time()
                    })
                    
                    display_results(results)
                    
                except Exception as e:
                    st.error(f"Error processing task: {str(e)}")
        else:
            st.warning("Please enter a task description.")

def display_results(results: Dict[str, Any]):
    """Display task results in an organized way"""
    st.header("Results")
    
    # Main results tabs
    tabs = st.tabs(["Summary", "Details", "Analysis", "Visualization"])
    
    with tabs[0]:  # Summary
        st.subheader("Task Summary")
        if 'web_results' in results:
            st.info(results['web_results'].get('summary', 'No summary available'))
        
        st.subheader("Key Findings")
        for pattern in results.get('patterns', []):
            st.write(f"- {pattern.description}")
    
    with tabs[1]:  # Details
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Generated Rules")
            for rule in results.get('rules', []):
                st.code(rule)
        
        with col2:
            st.subheader("System Actions")
            for action in results.get('actions', []):
                st.write(f"- {action}")
    
    with tabs[2]:  # Analysis
        st.subheader("System Analysis")
        st.json(results.get('analysis', {}))
        
        st.subheader("Complexity Layers")
        for layer in results.get('complexity_layers', []):
            with st.expander(f"Layer: {layer['type']}"):
                st.write(f"Complexity: {layer['complexity']}")
                st.write("Properties:", layer['properties'])
    
    with tabs[3]:  # Visualization
        st.subheader("Pattern Network")
        # Create network visualization using plotly
        if 'patterns' in results:
            pattern_fig = create_pattern_network(results['patterns'])
            st.plotly_chart(pattern_fig, use_container_width=True)

def create_pattern_network(patterns: List[Dict[str, Any]]) -> go.Figure:
    """Create a network visualization of patterns"""
    if not patterns:
        return go.Figure()
        
    # Create networkx graph
    G = nx.Graph()
    
    # Add nodes
    for i, pattern in enumerate(patterns):
        G.add_node(i, 
                  label=pattern['pattern_type'],
                  confidence=pattern.get('confidence', 0.5))
    
    # Add edges between related patterns
    for i, pattern in enumerate(patterns):
        for j in range(i):
            if patterns[j]['pattern_type'] in pattern.get('related_patterns', []):
                G.add_edge(i, j)
    
    # Calculate layout
    pos = nx.spring_layout(G)
    
    # Extract node positions
    node_x = []
    node_y = []
    node_sizes = []
    node_labels = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_sizes.append(G.nodes[node]['confidence'] * 40)
        node_labels.append(G.nodes[node]['label'])
    
    # Create edges traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Create figure
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_labels,
        textposition="top center",
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=node_sizes,
            line_width=2
        )
    ))
    
    # Update layout
    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    return fig

def render_history():
    """Render task history interface"""
    st.header("Task History")
    
    if not st.session_state.task_history:
        st.info("No tasks processed yet.")
        return
    
    # Create history dataframe
    history_df = pd.DataFrame(st.session_state.task_history)
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        task_type_filter = st.multiselect(
            "Filter by Task Type",
            options=history_df['type'].unique()
        )
    with col2:
        date_range = st.date_input(
            "Date Range",
            value=[history_df['timestamp'].min(), history_df['timestamp'].max()]
        )
    
    # Apply filters
    if task_type_filter:
        history_df = history_df[history_df['type'].isin(task_type_filter)]
    
    # Display filtered history
    st.dataframe(
        history_df[['timestamp', 'type', 'task']],
        use_container_width=True
    )
    
    # Task details
    if selected_task := st.selectbox("Select task to view details", history_df['task']):
        task_data = history_df[history_df['task'] == selected_task].iloc[0]
        display_results(task_data['results'])

def main():
    """Main application entry point"""
    initialize_session_state()
    render_sidebar()
    
    # Main content tabs
    tab1, tab2 = st.tabs(["Task", "History"])
    
    with tab1:
        render_task_interface()
    
    with tab2:
        render_history()

if __name__ == "__main__":
    main()
=======
# interface/streamlit_app.py

import streamlit as st
import asyncio
from typing import Dict
import logging
from datetime import datetime

from enhanced_learning_system import EnhancedLearningSystem

def create_interface():
    st.title("Alice - Enhanced Learning System")
    
    # Initialize system
    if 'system' not in st.session_state:
        st.session_state.system = EnhancedLearningSystem()
    
    # System state display
    st.sidebar.header("System State")
    system_state = st.session_state.system.system_state.get_summary()
    st.sidebar.json(system_state)
    
    # Quantum state display
    st.sidebar.header("Quantum State")
    quantum_state = st.session_state.system.consciousness.quantum_state
    st.sidebar.json(quantum_state)
    
    # Task input
    st.header("Task Input")
    task = st.text_area("Enter task description:")
    
    if st.button("Process Task"):
        if task:
            with st.spinner("Processing task..."):
                results = asyncio.run(st.session_state.system.process_web_task(task))
                
                # Display results
                st.header("Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Web Interaction Results")
                    st.json(results['web_results'])
                    
                    st.subheader("Detected Patterns")
                    st.write(results['patterns'])
                    
                with col2:
                    st.subheader("Generated Rules")
                    for rule in results['rules']:
                        st.write(f"- {rule}")
                    
                    st.subheader("System Analysis")
                    st.write(results['analysis'])
                
                # Display complexity layers
                st.header("Complexity Analysis")
                for layer in st.session_state.system.complexity_system.layers:
                    st.subheader(f"Layer: {layer.layer_type}")
                    st.write(f"Complexity Score: {layer.complexity_score}")
                    st.write(f"Emergent Properties: {layer.emergent_properties}")
        else:
            st.warning("Please enter a task description.")


>>>>>>> f4e70f2f7842b291c7366f2b1a38129785624fb1
