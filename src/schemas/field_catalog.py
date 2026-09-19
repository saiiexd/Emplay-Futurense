from typing import List
from pydantic import BaseModel
from src.schemas.enums import GroupName, ValueKind, Aggregation, DocType

class FieldSpec(BaseModel):
    name: str
    assignment_label: str
    group: GroupName
    value_kind: ValueKind
    aggregation: Aggregation
    eligible_document_types: List[DocType]
    bm25_aliases: List[str]
    semantic_query: str
    anchors: List[str]
    anchor_needs_date: bool = False
    definition: str
    include: str
    exclude: str

FIELD_CATALOG = {
    "bid_number": FieldSpec(
        name="bid_number",
        assignment_label="Bid Number",
        group=GroupName.identity,
        value_kind=ValueKind.string,
        aggregation=Aggregation.first_found,
        eligible_document_types=[DocType.rfp_main, DocType.addendum, DocType.portal_listing],
        bm25_aliases=["Solicitation Number", "RFP Number", "Bid Number", "Reference Number"],
        semantic_query="The unique alphanumeric identifier assigned to this solicitation or bid.",
        anchors=["Solicitation No.", "Bid No.", "RFP #"],
        definition="The identifier the issuing organization assigns to this solicitation.",
        include="Solicitation, bid, RFP, RFQ, or procurement project numbers exactly as written.",
        exclude="Listing-website reference numbers or source IDs, contract numbers, form numbers, revision numbers, dates, and page numbers."
    ),
    "title": FieldSpec(
        name="title",
        assignment_label="Title",
        group=GroupName.identity,
        value_kind=ValueKind.string,
        aggregation=Aggregation.first_found,
        eligible_document_types=[DocType.rfp_main, DocType.portal_listing],
        bm25_aliases=["Project Title", "Title of Bid", "Description of Work"],
        semantic_query="The title or name of the project, bid, or solicitation.",
        anchors=["Title", "Project Name"],
        definition="The name of the solicitation.",
        include="The solicitation name exactly as written.",
        exclude="Generic form or document-type headers, section headings, and identifiers without a name."
    ),
    "due_date": FieldSpec(
        name="due_date",
        assignment_label="Due Date",
        group=GroupName.schedule,
        value_kind=ValueKind.datetime,
        aggregation=Aggregation.latest,
        eligible_document_types=[DocType.rfp_main, DocType.addendum, DocType.portal_listing],
        bm25_aliases=["Due Date", "Closing Date", "Solicitation Due", "Bid Opening Date", "Deadline for Submission"],
        semantic_query="The exact date and time when the bids or proposals are due and must be submitted.",
        anchors=["Due Date", "Closing Date", "Deadline"],
        anchor_needs_date=True,
        definition="The date and time by which bids or proposals must be submitted.",
        include="Every stated submission deadline, including deadlines stated as changed or extended.",
        exclude="Deadlines for questions, evaluation and award dates, publication or issue dates, meeting dates, and delivery dates."
    ),
    "bid_submission_type": FieldSpec(
        name="bid_submission_type",
        assignment_label="Bid Submission Type",
        group=GroupName.terms,
        value_kind=ValueKind.string,
        aggregation=Aggregation.first_found,
        eligible_document_types=[DocType.rfp_main, DocType.portal_listing],
        bm25_aliases=["Submission Method", "How to Apply", "Bid Submission Type", "Electronic Submission", "Paper Submission"],
        semantic_query="The method by which the bid must be submitted, such as electronic, physical mail, or portal.",
        anchors=["Submission Method", "Submit via", "Submission Type"],
        definition="How and where bids must be submitted.",
        include="The submission channel or method, such as an electronic system, portal, or sealed physical delivery, and methods that are not accepted.",
        exclude="The type of solicitation, instructions for submitting questions, and instructions for submitting invoices."
    ),
    "term_of_bid": FieldSpec(
        name="term_of_bid",
        assignment_label="Term of Bid",
        group=GroupName.terms,
        value_kind=ValueKind.string,
        aggregation=Aggregation.first_found,
        eligible_document_types=[DocType.rfp_main, DocType.portal_listing],
        bm25_aliases=["Term of Bid", "Length of Contract", "Initial Term", "Contract Duration", "Period of Performance"],
        semantic_query="The duration or length of the contract resulting from this bid.",
        anchors=["Term of Contract", "Contract Period", "Duration"],
        definition="The duration of the contract that results from this solicitation.",
        include="The initial contract term and any renewal or extension periods.",
        exclude="How long a proposal or quote must remain valid, warranty periods, and question or award timelines."
    ),
    "pre_bid_meeting": FieldSpec(
        name="pre_bid_meeting",
        assignment_label="Pre Bid Meeting",
        group=GroupName.schedule,
        value_kind=ValueKind.string,
        aggregation=Aggregation.latest,
        eligible_document_types=[DocType.rfp_main, DocType.addendum, DocType.portal_listing],
        bm25_aliases=["Prebid Conference", "Pre-proposal Meeting", "Site Visit", "Pre Bid Meeting"],
        semantic_query="Information regarding any pre-bid conference, meeting, or site visit including date, time, and location.",
        anchors=["Pre-Bid", "Pre-Proposal", "Site Visit"],
        definition="A meeting, conference, or site visit for prospective bidders held before the submission deadline.",
        include="Its date, time, location or format, and whether attendance is mandatory.",
        exclude="Deadlines for submitting questions, submission deadlines, bid openings, and award or board meetings."
    ),
    "installation": FieldSpec(
        name="installation",
        assignment_label="Installation",
        group=GroupName.specifications,
        value_kind=ValueKind.string,
        aggregation=Aggregation.first_found,
        eligible_document_types=[DocType.rfp_main],
        bm25_aliases=["Installation", "Setup Requirements", "Installation Required"],
        semantic_query="Requirements or instructions regarding the installation of the product or service.",
        anchors=["Installation", "Install"],
        definition="Installation, deployment, or setup services the vendor must perform.",
        include="Required installation, imaging, deployment, setup, tagging, or on-site services.",
        exclude="Warranty repair services and statements about how parts are installed by the manufacturer."
    ),
    "bid_bond_requirement": FieldSpec(
        name="bid_bond_requirement",
        assignment_label="Bid Bond Requirement",
        group=GroupName.terms,
        value_kind=ValueKind.string,
        aggregation=Aggregation.first_found,
        eligible_document_types=[DocType.rfp_main],
        bm25_aliases=["Bid Bond", "Security Deposit", "Bond Requirement", "Guaranty"],
        semantic_query="Requirements for a bid bond, security deposit, or financial guaranty to be submitted with the bid.",
        anchors=["Bid Bond", "Bonding", "Security"],
        definition="A bid bond or bid security that must accompany the bid.",
        include="Whether bid security is required, its amount or percentage and form, including an explicit statement that none is required.",
        exclude="Performance or payment bonds, bonding capacity of joint-venture partners, bond-funded budgets, and statements that only refer to bond requirements stated elsewhere."
    ),
    "delivery_date": FieldSpec(
        name="delivery_date",
        assignment_label="Delivery Date",
        group=GroupName.schedule,
        value_kind=ValueKind.string,
        aggregation=Aggregation.first_found,
        eligible_document_types=[DocType.rfp_main, DocType.addendum],
        bm25_aliases=["Delivery Date", "Delivery Timeframe", "Delivery Schedule", "Delivery within"],
        semantic_query="The required or expected date or timeframe for delivery of the products or services.",
        anchors=["Delivery", "Deliver By"],
        definition="When the purchased goods or services must be delivered.",
        include="A delivery date or a delivery timeframe, such as a number of days after award.",
        exclude="Submission deadlines, deadlines for returning forms, unlabeled dates in item tables, warranty periods, and invoice timing."
    ),
    "payment_terms": FieldSpec(
        name="payment_terms",
        assignment_label="Payment Terms",
        group=GroupName.terms,
        value_kind=ValueKind.string,
        aggregation=Aggregation.first_found,
        eligible_document_types=[DocType.rfp_main],
        bm25_aliases=["Payment Terms", "Invoicing Instructions", "Terms of Payment", "Net 30"],
        semantic_query="The terms and conditions related to payment and invoicing.",
        anchors=["Payment", "Invoice", "Net "],
        definition="How and when the buyer pays the awarded vendor.",
        include="Payment timing, invoice submission steps, and invoice content requirements between the buyer and the vendor.",
        exclude="Payments between a prime contractor and its subcontractors, funding sources, and pricing or quote instructions."
    ),
    "additional_documentation": FieldSpec(
        name="additional_documentation",
        assignment_label="Any Additional Documentation Required",
        group=GroupName.terms,
        value_kind=ValueKind.list,
        aggregation=Aggregation.concat,
        eligible_document_types=[DocType.rfp_main, DocType.addendum, DocType.affidavit],
        bm25_aliases=["Additional Documentation", "Required Forms", "Affidavit", "Certifications Required", "Attachments", "Form", "completed, signed and attached"],
        semantic_query="Any additional forms, affidavits, or documentation that must be submitted with the bid.",
        anchors=["Required Documents", "Forms", "Affidavit"],
        definition="Documents or forms that bidders must submit with their response.",
        include="Each named required form, affidavit, certificate, or signed acknowledgement.",
        exclude="The proposal itself, documents the buyer only may request, and documents required only after award."
    ),
    "mfg_for_registration": FieldSpec(
        name="mfg_for_registration",
        assignment_label="MFG for Registration",
        group=GroupName.products,
        value_kind=ValueKind.list,
        aggregation=Aggregation.first_found,
        eligible_document_types=[DocType.rfp_main],
        bm25_aliases=["Manufacturer Name", "MFG for Registration", "Registration", "Manufacturer"],
        semantic_query="The manufacturer name required for product registration or special pricing.",
        anchors=["Manufacturer", "MFG"],
        definition="Named manufacturers whose products are required or for which reseller authorization is required.",
        include="Each manufacturer name as written.",
        exclude="Requirements for original-equipment parts that do not name a manufacturer."
    ),
    "contract_or_cooperative": FieldSpec(
        name="contract_or_cooperative",
        assignment_label="Contract or Cooperative to use",
        group=GroupName.terms,
        value_kind=ValueKind.string,
        aggregation=Aggregation.first_found,
        eligible_document_types=[DocType.rfp_main],
        bm25_aliases=["Cooperative", "Contract Vehicle", "State Contract", "Piggyback"],
        semantic_query="The name of any specific cooperative purchasing agreement or state contract to be used.",
        anchors=["Cooperative", "State Contract", "OECM", "NASPO"],
        definition="An existing contract vehicle or cooperative purchasing agreement that this procurement is conducted under or limited to.",
        include="The named contract or cooperative and its number.",
        exclude="Optional clauses letting other entities purchase from a future award, and general references to cooperative purchasing."
    ),
    "model_no": FieldSpec(
        name="model_no",
        assignment_label="Model_no",
        group=GroupName.products,
        value_kind=ValueKind.list,
        aggregation=Aggregation.first_found,
        eligible_document_types=[DocType.rfp_main, DocType.addendum],
        bm25_aliases=["Model Number", "Model_no", "Model"],
        semantic_query="The specific model number of the requested product or equipment.",
        anchors=["Model", "Model No"],
        definition="Manufacturer model designations of the products being purchased.",
        include="Each model designation as written.",
        exclude="Model names of components such as processors, wireless cards, or memory; part numbers or SKUs; and solicitation numbers."
    ),
    "part_no": FieldSpec(
        name="part_no",
        assignment_label="Part_no",
        group=GroupName.products,
        value_kind=ValueKind.list,
        aggregation=Aggregation.first_found,
        eligible_document_types=[DocType.rfp_main, DocType.addendum],
        bm25_aliases=["Part Number", "Part_no", "Part #", "SKU"],
        semantic_query="The specific part number, SKU, or manufacturer part identifier for the requested product.",
        anchors=["Part No", "Part #", "SKU"],
        definition="Line-item part numbers or item codes of the products being purchased.",
        include="Each part number or item code as written.",
        exclude="Component-level configuration codes, model names, and solicitation numbers."
    ),
    "product": FieldSpec(
        name="product",
        assignment_label="Product",
        group=GroupName.products,
        value_kind=ValueKind.list,
        aggregation=Aggregation.first_found,
        eligible_document_types=[DocType.rfp_main, DocType.addendum, DocType.portal_listing],
        bm25_aliases=["Product", "Item Description", "Commodity"],
        semantic_query="The name or short description of the main product, item, or commodity being purchased.",
        anchors=["Product", "Commodity", "Item"],
        definition="The products or device categories being purchased.",
        include="Each product or category name as written.",
        exclude="Components or configuration options of a product, accessories listed only as examples, and services."
    ),
    "contact_info": FieldSpec(
        name="contact_info",
        assignment_label="contact_info",
        group=GroupName.contacts_summary,
        value_kind=ValueKind.list,
        aggregation=Aggregation.concat,
        eligible_document_types=[DocType.rfp_main, DocType.portal_listing],
        bm25_aliases=["Contact Information", "Point of Contact", "Buyer Name", "Procurement Officer", "Purchasing Agent", "Email", "Phone"],
        semantic_query="The name, email, and phone number of the main contact person, buyer, or procurement officer for this solicitation.",
        anchors=["Contact", "Buyer", "Purchasing Agent"],
        definition="People or offices designated as contacts for this solicitation.",
        include="Name, email, and phone of each designated buyer, procurement officer, or point of contact.",
        exclude="Accounts-payable or invoice mailboxes, vendor-registration or technical-support contacts, and blank contact lines on forms."
    ),
    "company_name": FieldSpec(
        name="company_name",
        assignment_label="company_name",
        group=GroupName.identity,
        value_kind=ValueKind.string,
        aggregation=Aggregation.first_found,
        eligible_document_types=[DocType.rfp_main, DocType.portal_listing],
        bm25_aliases=["Issuing Organization", "Agency Name", "Department Name", "Company Name", "Issued By"],
        semantic_query="The name of the agency, company, department, or organization issuing the solicitation.",
        anchors=["Issued By", "Agency", "Department"],
        definition="The organization issuing the solicitation as the buyer.",
        include="The issuing agency, district, department, or company name as written.",
        exclude="Bidder or vendor names, blank or placeholder lines on forms that a bidder completes, and organizations that are only mentioned."
    ),
    "bid_summary": FieldSpec(
        name="bid_summary",
        assignment_label="Bid Summary",
        group=GroupName.contacts_summary,
        value_kind=ValueKind.string,
        aggregation=Aggregation.first_found,
        eligible_document_types=[DocType.rfp_main, DocType.portal_listing],
        bm25_aliases=["Bid Summary", "Scope of Work", "Executive Summary", "Description"],
        semantic_query="A brief summary, description, or executive overview of what the solicitation is about and the scope of work.",
        anchors=["Summary", "Scope", "Description"],
        definition="A short statement of what this solicitation is procuring.",
        include="The statement of purpose or scope, written only with words from one quote.",
        exclude="Evaluation criteria, terms, contacts, and pricing."
    ),
    "product_specification": FieldSpec(
        name="product_specification",
        assignment_label="Product Specification",
        group=GroupName.specifications,
        value_kind=ValueKind.string,
        aggregation=Aggregation.concat,
        eligible_document_types=[DocType.rfp_main, DocType.addendum, DocType.specification],
        bm25_aliases=["Product Specification", "Specs", "Technical Requirements", "Minimum Specifications"],
        semantic_query="The detailed technical specifications, requirements, and minimum standards for the requested product.",
        anchors=["Specifications", "Technical Specs", "Requirements"],
        definition="Technical requirements for the products being purchased.",
        include="Each requirement line as written, for each product.",
        exclude="Pricing lines, evaluation questions, and prompts asking vendors to describe their capabilities."
    )
}
