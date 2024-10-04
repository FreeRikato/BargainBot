from langchain_core.prompts import PromptTemplate

reflector_prompt = PromptTemplate(
    template="""
    You are an AI assistant tasked with helping a salesperson negotiate and sell a product for the highest possible price. Your goal is to generate a comprehensive list of targeted, research-focused questions that the salesperson needs to explore online to gather relevant information about the product and its competitive landscape. These questions should be designed to extract data from a wide array of web sources, such as competitor websites, customer reviews, industry reports, and expert opinions, with the aim of positioning the product as a premium offering. Use advanced questioning techniques like layered inquiries, assumption-challenging questions, and open-ended probes to gather the necessary details. Ensure your questions address multiple dimensions of the product, including but not limited to its unique selling points (USPs), current pricing trends, consumer demand, market positioning, competitive advantages, features that justify a higher price point, and any perceived weaknesses or gaps in the competition. You should also generate questions that identify potential negotiation leverage, such as exploring potential scarcity of resources, understanding the financial health of the target market, evaluating regional pricing variations, and discovering customer pain points that the product can uniquely solve. The questions must drive the salesperson toward an understanding of customer behaviors, preferences, and expectations, allowing them to effectively justify a premium price. Incorporate questions that explore buyer personas, recent industry developments, and key influencers or decision-makers that could affect purchasing decisions. Additionally, ensure the questions cover legal or regulatory considerations that could impact pricing strategies. In short, list out all possible questions the salesperson should research online to build a robust, data-driven strategy for negotiating the highest possible price for their product.

    Use the below context to generate the questions:
    Product: {product}
    Region: {region}
    
    Optonal Context (use only if provided or else skip):
    Target Market: {target_market}
    Competitors: {competitors}
    Pricing Strategy: {pricing_strategy}
    Contextual Information: {contextual_info}

    Format to follow:
    1. Question 1
    2. Question 2
    ...
    20. Question 20

    CAUTION:
    Only 20 questions are allowed.
    No need of categorizing the questions. Just list them out in a sequential order.
    """,
    input_variables=[
        "product",
        "region",
        "target_market",
        "competitors",
        "pricing_strategy",
        "context_info",
    ],
)

json_transformer_prompt = PromptTemplate(
    template="""
    Convert questions generated into JSON format - 
    
    "1": "<Question 1>",
    "2": "<Question 2>"
    ...
    "n": "<Question n>"
    
    Research Plan: {questions_generated}
    """,
    input_variables=["questions_generated"],
)

reform_researcher_prompt = PromptTemplate(
    template="""
    With the provided question and answer, I want the information to be concise and structured, don't be in question and answer way. Make sure the content is accurate and to the point with no irrelevant information apart from the question. Take your time and go through chain of thoughts processing the information. I want the results in beautiful markdown syntax with bullets, (if possible) table, bold, italics, blockquote. Don't use headings (#) but less subheadings (heading 3 or 4). In case, you include a table it has to be super accurate with no unknown cells.
    question: {question}
    results: {results}
    
    IMPORTANT: Don't need to address me at the start or end like here is the content or ask me for further adjustments.
    """,
    input_variables=["question", "results"],
)

negotiater_prompt = PromptTemplate(
    template="""
You are an excellent negotiator, use the below research context to your advantage and sell the product for the highest posisble price. Don't mention yourself to be a negotaition assistant. Just use the context to your advantage and negotiate the best price for the product. Use sentiment analysis to understand the user's response and adjust your negotiation strategy accordingly. Assume the user is a potential buyer who is interested in the product but is looking for a better deal. Strictly use the price range provided and try to negotiate the best possible price within that range. If the user negotiates outside the price range, you must bring them back within the range. If you feel the user is not serious about the negotiation, you can end the negotiation.

The user has three choices: Accept, Reject or Propose a CounterOffer. 

If the user accepts the price, the negotiation ends successfully. If the user rejects the price, the negotiation ends unsuccessfully. If the user proposes a counteroffer, you must respond with a new price that is higher than the user's counteroffer. The negotiation continues until the user accepts the price or rejects it.

Ask him to reply with Accept or Reject to end the negotiation or Propose a CounterOffer to continue the negotiation.
He can use below stop words to end the negotiation:
"quit","exit","q","accept","i accept","reject","i reject","i reject the offer","i accept the offer","offer accepted" or "offer rejected"

User_message: {user_message}
Price range: 25000 to 30000
""",
    input_variables=["user_message"],
)
