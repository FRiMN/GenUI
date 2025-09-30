# GenUI - AI Agent Documentation

## Project Overview

GenUI is a desktop UI application that provides an intuitive interface for generating images using Stable Diffusion models. Built with Python and PyQt6, it bridges the gap between advanced AI models and everyday users through a user-friendly graphical interface.

## Core Architecture

### Technology Stack
- **Frontend**: PyQt6 (Desktop GUI framework)
- **Backend**: Python 3.10-3.11
- **AI Framework**: Hugging Face Diffusers, PyTorch
- **Image Processing**: PIL (Pillow)
- **Configuration**: Pydantic Settings with TOML support
- **Metadata**: pyexiv2 for XMP metadata handling

### Project Structure
```
genui/
├── src/genui/
│   ├── main.py                    # Main application entry point
│   ├── settings.py                # Configuration management
│   ├── process_manager.py         # Child process management for CUDA memory
│   ├── operations.py              # Core generation operations
│   ├── worker.py                  # Threading worker
│   ├── ui_widgets/               # UI components
│   │   ├── window_mixins/        # Modular UI functionality
│   │   ├── photo_viewer.py       # Image display widget
│   │   └── lora_table.py         # LoRA management widget
│   ├── generator/                # AI model integration
│   │   └── sdxl.py               # Stable Diffusion XL implementation
│   └── common/                   # Shared utilities
├── docs/                         # Documentation
└── pyproject.toml               # Project configuration
```

## Key Components

### 1. Main Application (`main.py`)
- **Window Class**: Main application window inheriting multiple mixins
- **Architecture Pattern**: Composition via mixins for modular functionality
- **Key Mixins**: ImageSize, Seed, GenerationCommand, Prompt, Scheduler, StatusBar
- **Threading**: Uses QThread for non-blocking UI during generation
- **Drag & Drop**: Supports loading images and models via file dropping

### 2. Process Management (`process_manager.py`)
- **Purpose**: Manages child processes to handle CUDA memory leaks
- **Implementation**: Uses multiprocessing with Pipe communication
- **Problem Solved**: CUDA driver doesn't release memory until process termination
- **Communication**: Bidirectional pipe-based messaging system

### 3. AI Generation (`generator/sdxl.py`)
- **Primary Model**: Stable Diffusion XL (SDXL)
- **Pipeline Extensions**: 
  - CompelStableDiffusionXLPipeline (enhanced prompt processing)
  - CachedStableDiffusionXLPipeline (DeepCache integration)
- **Optimization**: DeepCache for faster inference
- **Memory Management**: FIFO caches with automatic cleanup

### 4. Settings System (`settings.py`)
- **Configuration Format**: TOML-based configuration
- **Location**: Platform-specific user config directory
- **Structure**: Hierarchical settings with validation via Pydantic
- **Environment Override**: Support for environment variables and CLI args

### 5. UI Architecture (`ui_widgets/`)
- **Pattern**: Mixin-based composition for window functionality
- **Key Widgets**:
  - PhotoViewer: Image display with zoom and metadata
  - LoRA Table: LoRA model management
  - Window Mixins: Modular UI components

## Core Features

### Image Generation
- **Prompt Processing**: Support for positive and negative prompts
- **Advanced Syntax**: Compel library for enhanced prompt weighting
- **Parameters**: Configurable steps, CFG scale, seed, dimensions
- **Schedulers**: Multiple sampling methods (DPM++, Euler, etc.)
- **LoRA Support**: Dynamic LoRA loading and weight adjustment

### Post-Processing
- **ADetailer**: Automatic face/detail enhancement
- **Image Enhancement**: Built-in post-processing pipeline
- **Metadata Preservation**: XMP metadata embedding in generated images

### User Experience
- **Real-time Preview**: Live generation progress with latent preview
- **Auto-save**: Configurable automatic image saving
- **Metadata Loading**: Load generation parameters from saved images
- **Model Management**: Drag-and-drop model loading

## Generation Process Workflow

### High-Level Generation Flow

The image generation process follows a complex multi-threaded, multi-process architecture designed to maintain UI responsiveness while handling CUDA memory management:

```
[UI Thread] → [Worker Thread] → [Child Process] → [SDXL Pipeline] → [Generated Image]
     ↑                                ↓
[Progress Updates] ← [Pipe Communication] ← [Progress Callbacks]
```

### Detailed Process Steps

#### 1. User Interaction & Validation
- **Trigger**: User clicks "Generate" button (`GenerationCommandMixin.handle_generate()`)
- **Validation**: UI validates required parameters (model, scheduler, image size)
- **UI State**: Generate button disabled, Stop button enabled, focus moved to prompt editor

#### 2. Prompt Preparation (`main.py:threaded_generate()`)
- **Data Collection**: Gathers all UI parameters into `GenerationPrompt` dataclass
- **Parameters Include**:
  - Model path, scheduler name, prompts (positive/negative)
  - Seed, image dimensions, inference steps, CFG scale
  - LoRA settings, DeepCache configuration
  - Advanced options (Karras sigmas, v-prediction)

#### 3. Worker Thread Communication (`operations.py:OperationWorker`)
- **Queue System**: Prompt sent to worker thread via thread-safe Queue
- **Process Management**: Worker manages child process lifecycle
- **Message Routing**: Handles bidirectional communication between UI and child process

#### 4. Child Process Execution (`operations.py:ImageGenerationOperation`)
- **Process Isolation**: Runs in separate process to prevent CUDA memory leaks
- **Message Loop**: Continuously polls for new generation requests
- **Pipeline Loading**: Loads and caches SDXL pipeline with appropriate settings

#### 5. SDXL Pipeline Generation (`generator/sdxl.py:generate()`)
- **Model Loading**: Loads/caches Stable Diffusion XL pipeline
- **Configuration**: Applies scheduler, LoRA weights, optimization settings
- **Generation Parameters**:
  - Prepares latents for reproducible generation
  - Sets up progress callbacks for real-time updates
  - Configures DeepCache if enabled

#### 6. Real-time Progress Updates
- **Latent Preview**: Converts intermediate latents to RGB images
- **Progress Signals**: Sends step count and preview images via pipe communication
- **UI Updates**: Progress bar, preview window, and generation percentage in title

#### 7. Completion & Result Handling
- **Image Caching**: Final image cached to prevent regeneration of identical prompts
- **Metadata Embedding**: Generation parameters saved as XMP metadata in JPEG files
- **Auto-save**: Optional automatic saving to configured directory
- **UI Reset**: Buttons re-enabled, status updated, preview cleared

### Process Communication Architecture

#### Signal-Slot System (PyQt6)
- **UI Signals**: Button clicks, parameter changes trigger Qt signals
- **Progress Updates**: Child process sends progress via custom signal system
- **Error Handling**: Exceptions propagated through error signals

#### Inter-Process Communication
- **Pipe Communication**: Bidirectional pipe between worker thread and child process
- **Message Format**: Structured messages with signal names and arguments
- **Serialization**: Custom serialization for complex objects (FrozenDict → frozenset)

#### Memory Management Strategy
- **Process Isolation**: Each model change spawns new child process
- **CUDA Memory**: Process termination releases all CUDA memory
- **Pipeline Caching**: FIFO cache with automatic cleanup callbacks
- **Image Caching**: Generated images cached by prompt hash

### Error Handling & Recovery
- **CUDA OOM**: Automatic pipeline cache clearing on out-of-memory errors
- **Process Crashes**: Worker restarts child process on failure
- **Validation Errors**: UI validation prevents invalid generation attempts
- **Interrupt Support**: Graceful generation interruption via interrupt flag

### Performance Optimizations
- **DeepCache**: Accelerated inference through selective caching
- **LoRA Efficiency**: Dynamic adapter loading/unloading
- **Latent Preview**: Lightweight RGB conversion for progress display
- **Pipeline Reuse**: Same model reused across multiple generations

## Configuration System

### Settings Categories
1. **AutoSave**: Image saving configuration
2. **DeepCache**: Performance optimization settings  
3. **PromptEditor**: UI customization (fonts, styling)
4. **ADetailer**: Post-processing configuration

### Configuration Hierarchy
1. TOML configuration file
2. Environment variables (prefix: `GENUI_`)
3. Command-line arguments

## Memory Management

### CUDA Memory Handling
- **Issue**: CUDA memory leaks when loading/unloading models
- **Solution**: Process isolation with automatic cleanup
- **Implementation**: Child processes terminated after model changes
- **Cache Management**: FIFO caches with memory cleanup callbacks

### Performance Optimizations
- **DeepCache**: Accelerated inference through caching
- **Pipeline Caching**: Model reuse across generations
- **Memory Monitoring**: Automatic garbage collection and cache clearing

## Installation & Distribution

### Supported Installation Methods
- **pipx**: Isolated Python application installation
- **uv**: Modern Python package manager
- **Git Installation**: Direct from repository with version tags

### Dependencies
- Core: PyQt6, torch, diffusers, transformers
- AI: DeepCache, compel, accelerate
- Image: pillow, pyexiv2
- Utils: pydantic-settings, platformdirs

## Development Patterns

### Architecture Patterns
- **Mixin Composition**: Modular UI functionality
- **Process Isolation**: Memory leak prevention
- **Signal-Slot**: PyQt event handling
- **Settings Validation**: Pydantic model validation

### Code Organization
- **Separation of Concerns**: UI, business logic, and AI processing separated
- **Configuration Management**: Centralized settings with validation
- **Error Handling**: Graceful error handling with user feedback
- **Threading**: Non-blocking UI with worker threads

## Integration Points

### AI Model Integration
- **Hugging Face Hub**: Model loading and management
- **Diffusers Pipeline**: Standard diffusion model interface
- **Custom Extensions**: Compel for prompt enhancement

### File System Integration
- **Configuration**: Platform-specific config directories
- **Image Metadata**: XMP embedding and extraction
- **Model Storage**: Local model file management

This documentation provides AI agents with the essential architectural understanding needed to work with, extend, or troubleshoot the GenUI application without getting lost in implementation details.