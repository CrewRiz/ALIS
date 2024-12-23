# ALIS (Advanced Learning and Intelligence System)

ALIS is a sophisticated AI system that combines quantum-inspired computation, consciousness simulation, pattern recognition, and adaptive learning capabilities with Gödel-based novelty detection and Lamarckian evolution.

## Key Features

### Core Architecture
- **Quantum-Inspired Processing**: Simulates quantum effects and multiple potential futures using superposition concepts
- **Enhanced Learning System**: Core learning and decision-making engine
- **Gödel-Based Novelty Detection**: Identifies system limitations and triggers self-improvement
- **Pattern Detection**: Advanced pattern recognition using LLM capabilities
- **Lamarckian Rule Evolution**: Rules adapt based on experience
- **Safety First**: Comprehensive safety checks and logging

### Technical Capabilities
- Web automation and interaction
- Pattern detection and analysis
- Dynamic rule generation
- RAG-based memory system
- Safety-focused implementation

## Setup

1. Create a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with:
```
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

## Project Structure

- `enhanced_learning_system.py`: Core learning engine
- `quantum_consciousness.py`: Quantum-inspired processing
- `pattern_detection.py`: Pattern recognition system
- `safety.py`: Safety checks and logging
- `settings.py`: Global configuration
- `base_classes.py`: Core data structures
- `streamlit_app.py`: Web interface

## Usage

The system can be used through the Streamlit interface:
```bash
streamlit run streamlit_app.py
```

## Safety Features

- Command safety checks
- URL/domain validation
- Path access control
- Content sensitivity scanning
- Comprehensive logging
- Time-based action limits

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

See LICENSE.txt for details.
