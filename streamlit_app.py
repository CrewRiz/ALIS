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
        priority = st.select_slider(
            "Priority",
            options=["Low", "Medium", "High"],
            value="Medium"
        )
    
    # Advanced options
    with st.expander("Advanced Options"):
        col1, col2 = st.columns(2)
        with col1:
            complexity = st.slider(
                "Max Complexity",
                min_value=1,
                max_value=1000,
                value=SETTINGS['system']['max_complexity']
            )
        with col2:
            simulation_depth = st.slider(
                "Simulation Depth",
                min_value=1,
                max_value=10,
                value=SETTINGS['system']['max_simulation_depth']
            )
    
    # Submit button
    if st.button("Submit Task"):
        with st.spinner("Processing task..."):
            results = asyncio.run(st.session_state.system.process_task(
                task=task,
                task_type=task_type,
                priority=priority,
                max_complexity=complexity,
                simulation_depth=simulation_depth
            ))
            
            st.session_state.task_history.append({
                'task': task,
                'type': task_type,
                'results': results,
                'timestamp': SETTINGS.get_current_time()
            })
            
            display_results(results)

def display_results(results: Dict[str, Any]):
    """Display task results in an organized way"""
    st.subheader("Task Results")
    
    # Main results
    st.json(results['summary'])
    
    # Patterns found
    if 'patterns' in results:
        st.subheader("Patterns Detected")
        create_pattern_network(results['patterns'])
    
    # Actions taken
    if 'actions' in results:
        st.subheader("Actions Taken")
        actions_df = pd.DataFrame(results['actions'])
        st.dataframe(actions_df)
    
    # Quantum state changes
    if 'quantum_changes' in results:
        st.subheader("Quantum State Changes")
        before = results['quantum_changes']['before']
        after = results['quantum_changes']['after']
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Before")
            st.json(before)
        with col2:
            st.write("After")
            st.json(after)
    
    # Learning progress
    if 'learning' in results:
        st.subheader("Learning Progress")
        st.line_chart(results['learning']['progress'])

def create_pattern_network(patterns: List[Dict[str, Any]]):
    """Create a network visualization of patterns"""
    G = nx.Graph()
    
    # Add nodes
    for pattern in patterns:
        G.add_node(pattern['id'], 
                  type=pattern['type'],
                  confidence=pattern['confidence'])
    
    # Add edges based on relationships
    for pattern in patterns:
        if 'related_patterns' in pattern:
            for related in pattern['related_patterns']:
                if related in [p['id'] for p in patterns]:
                    G.add_edge(pattern['id'], 
                             related,
                             weight=pattern['relationships'][related])
    
    # Create positions
    pos = nx.spring_layout(G)
    
    # Create figure
    fig = go.Figure()
    
    # Add edges
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    ))
    
    # Add nodes
    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    node_text = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_colors.append(G.nodes[node]['confidence'])
        node_sizes.append(30)
        node_text.append(f"ID: {node}<br>Type: {G.nodes[node]['type']}<br>Confidence: {G.nodes[node]['confidence']:.2f}")
    
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=node_colors,
            size=node_sizes,
            colorbar=dict(
                thickness=15,
                title='Confidence',
                xanchor='left',
                titleside='right'
            ),
            line_width=2
        )
    ))
    
    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_history():
    """Render task history interface"""
    st.header("Task History")
    
    if not st.session_state.task_history:
        st.info("No tasks have been processed yet.")
        return
    
    # Convert history to dataframe
    history_data = []
    for item in st.session_state.task_history:
        history_data.append({
            'Timestamp': item['timestamp'],
            'Task': item['task'],
            'Type': item['type'],
            'Status': item['results']['status']
        })
    
    history_df = pd.DataFrame(history_data)
    history_df = history_df.sort_values('Timestamp', ascending=False)
    
    # Display history table
    st.dataframe(history_df)
    
    # Task type distribution
    st.subheader("Task Distribution")
    task_dist = history_df['Type'].value_counts()
    st.bar_chart(task_dist)
    
    # Success rate over time
    st.subheader("Success Rate Over Time")
    success_rate = (history_df['Status'] == 'success').rolling(10).mean()
    st.line_chart(success_rate)

def main():
    """Main application entry point"""
    initialize_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Main content tabs
    tab1, tab2 = st.tabs(["Task", "History"])
    
    with tab1:
        render_task_interface()
    with tab2:
        render_history()

if __name__ == "__main__":
    main()
