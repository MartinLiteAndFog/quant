# Other — live

# Expected Trades Data Format Documentation

## Overview
The `expected_trades.jsonl` file is a data storage format used to track anticipated trading actions in the system. This file follows a line-delimited JSON (JSONL) format, where each line represents a single expected trade record.

## Record Structure
Each trade record contains the following fields:

```typescript
{
  ts: string;           // ISO 8601 timestamp with timezone
  symbol: string;       // Trading pair (e.g., "SOL-USDT")
  side: string;         // Trade direction ("short" or "long")
  action: string;       // Trading action (e.g., "exit_flip")
  qty: number;         // Trade quantity
  expected_px: number; // Expected price for the trade
  client_oid: string | null;  // Client order ID (optional)
  signal_id: string | null;   // Associated signal ID (optional)
  note: string;        // Human-readable description of trade context
}
```

## Field Details

### Required Fields
- `ts`: Timestamp of when the trade is expected to execute
- `symbol`: The trading pair identifier using standard market notation
- `side`: Trading direction
- `action`: The type of trading action to be performed
- `qty`: Quantity to trade
- `expected_px`: Target price for the trade execution
- `note`: Contextual information about the trade

### Optional Fields
- `client_oid`: Optional identifier for client-side order tracking
- `signal_id`: Optional reference to an associated trading signal

## Notes Format
The `note` field follows a structured format that captures the trading context:
```
executor action={action_type} event={event_type} current={current_position}
```

Example:
```
"executor action=flip_to_short event=tp_exit current=long"
```

## Usage Context
This file serves as an interface between the trading strategy components and the execution system. It records expected trades before they are executed, allowing for:
- Trade intention logging
- Execution verification
- Strategy debugging
- Performance analysis

## File Format
- The file uses the `.jsonl` extension
- Each record is on a separate line
- UTF-8 encoding is assumed
- Records are typically ordered chronologically by timestamp

## Integration Points
This data format is typically:
- Written by strategy executors
- Read by trade execution systems
- Used by monitoring and analysis tools
- Referenced for trade reconciliation