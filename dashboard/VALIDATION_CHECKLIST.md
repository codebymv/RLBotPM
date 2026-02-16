# Dashboard Visual Language Overhaul - Validation Checklist

## Test Scenarios

### 1. Mode Toggle Functionality

#### Paper Mode (Default)
- [ ] Dashboard loads with Paper mode active by default
- [ ] URL shows `?mode=paper` after first interaction
- [ ] All KPI cards display paper trading data
- [ ] Mode toggle shows Paper button as active (amber highlight)
- [ ] StatusPill shows "PAPER" with amber styling

#### Live Mode
- [ ] Clicking Live mode button switches to live data
- [ ] URL updates to `?mode=live`
- [ ] All KPI cards display live trading data
- [ ] Mode toggle shows Live button as active (cyan highlight)
- [ ] StatusPill shows "LIVE" with cyan styling and pulse animation

#### Mode Persistence
- [ ] Mode selection persists when navigating between pages
- [ ] Refreshing page maintains selected mode via URL param
- [ ] Direct link with `?mode=live` correctly loads live mode
- [ ] Direct link with `?mode=paper` correctly loads paper mode

### 2. Empty State Scenarios

#### No Data - Paper Mode
- [ ] Overview page shows "No paper trading data available"
- [ ] Positions page shows "No open paper positions"
- [ ] Edge Health page shows "No settled paper trades yet"
- [ ] Empty states include helpful submessage text
- [ ] Empty state styling is consistent (gray border, centered)

#### No Data - Live Mode
- [ ] Overview page shows "No live trading data available"
- [ ] Positions page shows "No open live positions"
- [ ] Edge Health page shows "No settled live trades yet"
- [ ] Live mode empty states use cyan accent colors

#### No Trades in Mode
- [ ] Switching to a mode with zero trades shows appropriate empty state
- [ ] Recent trades table shows "No {mode} trades recorded"
- [ ] Side breakdown section hides gracefully when no data

### 3. Live = 0 Scenario (No Live Trading Yet)

- [ ] Live mode toggle is accessible and functional
- [ ] All live mode pages show $0.00 realized P&L
- [ ] Open positions count shows 0
- [ ] Win/Loss record shows 0W / 0L
- [ ] No confusion between paper simulation and real money
- [ ] Empty states clearly indicate "Trades will appear here once executed"

### 4. Live > 0 Scenario (Active Live Trading)

#### With Open Positions
- [ ] Positions are clearly grouped by asset
- [ ] Each position shows mode badge (PAPER vs LIVE)
- [ ] Live positions have distinct cyan border accent
- [ ] Cost deployed and max profit calculated correctly
- [ ] Spot prices display when available

#### With Trade History
- [ ] Recent trades table filters by mode correctly
- [ ] Mode column shows LIVE badge with cyan styling
- [ ] P&L displays with correct color coding (green/red)
- [ ] Live trades are visually distinct from paper trades

#### Performance Metrics
- [ ] Total P&L shows live-only data when in live mode
- [ ] Win rate calculation excludes paper trades
- [ ] Side breakdown (BUY_NO / BUY_YES) filters by mode
- [ ] Edge health metrics are mode-scoped

### 5. Visual Language Consistency

#### Typography
- [ ] IBM Plex Mono used for body text and metrics
- [ ] Space Mono used for headings
- [ ] All numbers use tabular-nums for alignment
- [ ] Font sizes are consistent across pages

#### Color Coding
- [ ] Paper mode: Amber (#f59e0b) highlights
- [ ] Live mode: Cyan (#22d3ee) highlights
- [ ] Positive P&L: Green (#4ade80)
- [ ] Negative P&L: Red (#f87171)
- [ ] BUY_NO: Green background
- [ ] BUY_YES: Red background
- [ ] Neutral elements: Gray tones

#### Spacing & Layout
- [ ] Consistent padding in cards (p-4 to p-6)
- [ ] Grid gaps are uniform (gap-3 to gap-6)
- [ ] Section headers have consistent margins
- [ ] Terminal grid background visible on all main pages

### 6. Accessibility

#### Keyboard Navigation
- [ ] Tab key navigates through all interactive elements
- [ ] Mode toggle buttons are keyboard accessible
- [ ] Navigation links respond to Enter key
- [ ] Focus indicators are visible (cyan outline)
- [ ] Tab order is logical (left-to-right, top-to-bottom)

#### Screen Reader Support
- [ ] Mode toggle has aria-label="Trading mode selector"
- [ ] Active mode button has aria-pressed="true"
- [ ] Navigation links have aria-current="page" when active
- [ ] KPI cards have descriptive aria-labels
- [ ] Empty states have aria-live="polite"

#### Touch Targets (Mobile)
- [ ] All buttons are minimum 44px height on mobile
- [ ] Mode toggle buttons are easily tappable
- [ ] Navigation links are touch-friendly
- [ ] No accidental clicks due to small targets

### 7. Mobile Responsiveness

#### Layout Adaptation
- [ ] Grid columns reduce on smaller screens (2-col → 1-col)
- [ ] Navigation wraps gracefully or scrolls horizontally
- [ ] KPI cards stack vertically on mobile
- [ ] Tables scroll horizontally without breaking layout
- [ ] No horizontal overflow on small screens

#### Text Sizing
- [ ] All text is readable without zooming
- [ ] Headers scale down appropriately
- [ ] Sublabels remain legible
- [ ] Font sizes meet WCAG standards (16px+ base)

### 8. Component Reusability

- [ ] KpiCard used consistently across pages
- [ ] StatusPill appears on all relevant pages
- [ ] SectionHeader provides uniform section breaks
- [ ] EmptyState styling is identical everywhere
- [ ] DataFreshness component shows update times

### 9. Data Freshness Indicators

- [ ] Last updated timestamp displays when available
- [ ] Stale data (>5 min) shows amber warning indicator
- [ ] Fresh data (<5 min) shows gray inactive indicator
- [ ] Relative time updates correctly (seconds/minutes/hours)

### 10. Cross-Page Navigation

- [ ] Mode selection persists across all pages
- [ ] Navigation highlights current page correctly
- [ ] Breadcrumb trail is clear (via page titles)
- [ ] Back button maintains mode state
- [ ] External links (if any) open in new tabs

## Performance Checks

- [ ] Pages load without layout shift
- [ ] Mode toggle response is instant
- [ ] No FOUC (flash of unstyled content)
- [ ] Fonts load without blocking render
- [ ] Images/icons load progressively

## Browser Compatibility

Test in:
- [ ] Chrome/Edge (latest)
- [ ] Firefox (latest)
- [ ] Safari (latest)
- [ ] Mobile Safari (iOS)
- [ ] Chrome Mobile (Android)

## Known Issues / Edge Cases

Document any discovered issues here:

---

## Sign-off

- [ ] All critical scenarios pass
- [ ] Accessibility requirements met (WCAG AA)
- [ ] Mobile experience is smooth
- [ ] Visual language is distinctive and consistent
- [ ] Paper vs Live separation is crystal clear

**Tester:** ___________  
**Date:** ___________  
**Build/Commit:** ___________
