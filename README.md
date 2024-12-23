# ALIS (Advanced Learning and Intelligence System)

ALIS is a sophisticated AI system that combines quantum-inspired computation, pattern recognition, and adaptive learning capabilities.

## Features

- **Enhanced Learning System**: Core learning and decision-making engine
- **Quantum-Inspired Processing**: Simulates quantum effects for enhanced decision making
- **Pattern Detection**: Advanced pattern recognition using LLM capabilities
- **Web Interaction**: Safe and controlled web interaction capabilities
- **Safety First**: Comprehensive safety checks and logging
- **Genetic Memory**: Adaptive learning with experience-based modification

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
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
- `web_interaction.py`: Web interaction capabilities
- `safety.py`: Safety checks and logging
- `settings.py`: Global configuration
- `genetic_memory.py`: Adaptive memory system
- `base_classes.py`: Core data structures

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
