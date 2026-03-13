# Other — plans

# Plans Module Documentation

## Overview

The Plans module contains approved design documents and implementation plans for major system components. These documents serve as both architectural records and detailed implementation guides.

## Key Documents

### 1. Regime-Aware Dashboard Design
**Location:** `docs/plans/2026-02-23-regime-dashboard-design.md`

Defines the architecture for persisting and visualizing regime state, including:
- Independent regime store with SQLite → Postgres migration path
- Multi-horizon confidence scoring model
- Dashboard visualization with gate spans, levels, and trades
- API contracts for regime state and transitions

### 2. Dashboard V2 Design 
**Location:** `docs/plans/2026-02-26-dashboard-v2-design.md`

Expands the dashboard with advanced visualization features:
- State space heatmaps showing system position in 3D feature space
- Continuous regime gradient band replacing binary shading
- Historical trajectory visualization
- Real-time axis status bars

### 3. Predictive Coding Engine Design
**Location:** `docs/plans/2026-02-27-predictive-coding-engine-design.md`

Specifies a predictive coding based trading engine:
- Linear temporal model with adaptive precision
- Multi-horizon price level forecasting
- Probability-driven trade decisions
- Comprehensive backtest infrastructure

### 4. Kraken Bot Rewrite Design
**Location:** `docs/plans/2026-03-03-kraken-bot-rewrite-design.md`

Details rewrite of the Kraken Futures bot to achieve backtest parity:
- State machine for position management
- Dual execution engines (flip vs TP2) based on gate state
- API integration with quant service
- State persistence and metrics

### 5. Live 3-Axis Gate Design
**Location:** `docs/plans/2026-03-03-live-3axis-gate-design.md`

Describes real-time gate implementation using state space features:
- Integration with dashboard state space pipeline
- 3-axis threshold logic
- Priority-based gate state resolution
- API endpoint for bot consumption

## Document Structure

Each design document follows a consistent pattern:
1. Header with date, status and scope
2. Goals and requirements
3. Architecture decisions with rationale
4. Detailed technical specifications
5. Implementation plan broken into discrete tasks
6. Testing strategy
7. Deployment considerations

Implementation plans include:
- File-by-file changes needed
- Test coverage requirements
- Step-by-step implementation sequence
- Expected challenges and mitigations

## Usage

These documents serve multiple purposes:

1. **Design Record**
- Captures key architectural decisions and rationale
- Documents system behavior specifications
- Provides context for code review

2. **Implementation Guide**
- Breaks complex changes into manageable tasks
- Defines clear acceptance criteria
- Ensures consistent implementation approach

3. **Maintenance Reference**
- Explains intended system behavior
- Documents key design constraints
- Helps debug issues by showing original intent

## Best Practices

When working with plan documents:

1. **Implementation**
- Follow tasks in sequence - they build on each other
- Run tests at each step as specified
- Commit with suggested messages for traceability

2. **Modifications**
- Update plans if significant deviations are needed
- Document why changes were required
- Ensure test coverage remains aligned

3. **New Plans**
- Follow established document structure
- Include clear tasks and acceptance criteria
- Consider impacts on existing components

## Integration Points

The plans connect to the codebase through:

- File paths matching source tree structure
- Test files aligned with implementation
- API contracts defining integration points
- Metrics and monitoring specifications

This tight coupling between plans and code helps maintain consistency as the system evolves.

## Summary

The Plans module provides the authoritative design documentation and implementation guidance for major system components. Following these plans ensures consistent, well-tested implementation of complex features while maintaining architectural integrity.