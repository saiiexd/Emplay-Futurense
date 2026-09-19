"""Expected values observed in the supplied corpus.

These literals live in the test fixtures on purpose. Production code must derive
its behaviour from document provenance and generic amendment language, never
from the answers below.
"""

# Bid1 - Dallas ISD, RFP JA-207652. Addendum No. 2 extends the due date.
BID1 = {
    "bid_id": "Bid1",
    "main_due_date_marker": "Solicitation Due",
    "main_due_date_value": "27-JUN-2024 14:00:00",
    "addendum_due_date_marker": "The new due date for this RFP will be",
    "addendum_due_date_value": "July 9, 2024 at 2:00 PM CST",
    "addendum_number_that_amends": 2,
    # Addendum No. 1 answers vendor questions; it clarifies, it does not amend.
    "addendum_qa_marker": "only require etching on Laptops",
    "solicitation_number": "JA-207652",
}

# Bid2 - Maryland State Treasurer's Office PORFP.
BID2 = {
    "bid_id": "Bid2",
    "solicitation_number": "BPM044557",
    "porfp_number": "E20P4600040",
    "delivery_marker": "Delivery within 45 days of Award",
    "master_contract": "060B5400007",
    "model_numbers": ["Latitude 5550", "WD22TB4"],
    "part_number": "CC7802",
    "manufacturer": "Dell",
    "mercury_affidavit": "Mercury Affidavit",
    "contact_email": "thawkins@treasurer.state.md.us",
}
