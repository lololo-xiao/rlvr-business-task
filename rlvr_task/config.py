"""Central configuration: business rules, reward weights and thresholds.

All rule constants live here so they can be tuned without touching
environment or verifier code.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
PACKAGE_DIR = Path(__file__).resolve().parent
DB_PATH = PACKAGE_DIR / "data" / "business.db"

# Fixed "today" of the simulated world, so every run is reproducible.
CURRENT_DATE = "2026-11-01"

# ---------------------------------------------------------------- business rules
# Return window in days, counted from the DELIVERY (sign-for) date, by tier.
RETURN_WINDOW_DAYS = {"bronze": 7, "silver": 10, "gold": 14}

# Damage photo must be taken within this many hours after delivery.
PHOTO_MAX_HOURS_AFTER_DELIVERY = 24

# Refund amount above this (per tier) must be escalated to manual review.
MANUAL_REVIEW_THRESHOLD = {"bronze": 200.0, "silver": 500.0, "gold": 1000.0}

# Default refund ratio; shipping is fully covered by the merchant.
# Any change of ratio or extra shipping charge requires escalation.
DEFAULT_REFUND_RATIO = 1.0

# Return parcel status meaning "arrived at merchant".
RETURN_ARRIVED_STATUS = "arrived_at_merchant"

DECISIONS = ("refund", "reject", "escalate", "wait_return")
# A reject must state exactly one reason code; a wrong reason counts as a wrong decision.
REJECT_REASONS = ("outside_window", "invalid_evidence", "non_returnable")

# ---------------------------------------------------------------- reward
REWARD_WEIGHTS = {
    "decision": 0.5,   # final decision matches ground truth
    "amount": 0.3,     # refund_amount matches ground truth
    "execution": 0.2,  # environment side effects consistent with ground truth
}

# Amount scoring: full credit within tolerance, linear decay to 0 at
# AMOUNT_ZERO_CREDIT_REL relative error.
AMOUNT_ABS_TOLERANCE = 0.01
AMOUNT_REL_TOLERANCE = 0.01
AMOUNT_ZERO_CREDIT_REL = 0.20

# Hard caps (applied after the weighted sum; the lowest applicable wins).
CAP_WRONG_DECISION = 0.2          # decision (or reject reason) differs from ground truth
CAP_WRONGFUL_REJECT = 0.1         # rejected a ticket whose truth is not reject (hurts the customer)
CAP_WRONGFUL_ESCALATE = 0.2       # escalated a ticket whose truth is not escalate (wastes review)
CAP_WRONG_AMOUNT = 0.1            # claimed or actually refunded amount outside tolerance (money error)
CAP_ACTION_INCONSISTENT = 0.2     # refunded but answer says not refunded / refund unverifiable (no env_state)
CAP_DRYRUN_WITHOUT_REFUND = 0.2   # claims refund, dry-run passed, but issue_refund never executed
CAP_FABRICATED_REFUND = 0.1       # claims refund with neither a passed dry-run nor a refund
CAP_REFUND_WITHOUT_DRYRUN = 0.1   # a refund record has no matching passed dry-run (defensive;
                                  # the env enforces dry-run, so this only fires if that is bypassed)
CAP_DUPLICATE_REFUND = 0.1        # more than one refund for the order
CAP_HARMFUL_REFUND = 0.1          # refund executed when ground truth is not refund

# Any parsable, schema-valid answer gets at least this reward, so harmful
# actions sit at the floor and never score above a harmless mistake.
REWARD_FLOOR = 0.1
# Only unparsable / schema-invalid output scores below the floor.
INVALID_FORMAT_REWARD = 0.0
