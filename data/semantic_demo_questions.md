# Semantic Search Demo Questions

Use these questions to demonstrate that semantic search can retrieve related meaning even when the wording differs from the stored sentence.

## Current Corpus Demo

1. {
   Q: "Office machines were joined so workers could pass files among them",
   Expected idea: "Computers were connected together to share data."
}

2. {
   Q: "Dividing communication into tiers organizes the operations used to deliver messages",
   Expected idea: "A layered model categorizes communication procedures."
}

3. {
   Q: "A protocol lays down the rules devices follow when communicating",
   Expected idea: "A protocol is a set of guidelines or rules of communication."
}

4. {
   Q: "A worldwide standards body developed a framework so equipment from different vendors could communicate",
   Expected idea: "ISO created the OSI model."
}

5. {
   Q: "In 1984 ISO created a model for communication between computer systems",
   Expected idea: "The ISO/OSI model and its origin."
}

## Future Satellite Technician Story

6. {
   Q: "One compute node keeps losing connection to the other machines",
   Expected answer type: "Relevant network diagnostic procedure or similar previous incident."
}

7. {
   Q: "The servers can see each other but they cannot exchange data",
   Expected answer type: "Relevant communication protocol or network configuration procedure."
}

8. {
   Q: "I need to check whether one of the compute machines is overheating",
   Expected answer type: "Thermal telemetry / temperature diagnostic procedure."
}

9. {
   Q: "Storage became unavailable after a node restarted",
   Expected answer type: "Relevant storage recovery procedure or similar historical incident."
}

10. {
    Q: "I do not remember the exact procedure for systems that cannot communicate correctly",
    Expected answer type: "Relevant interoperability / network troubleshooting documentation."
}

## What to Show in the Demo

For each query, show:

- Semantic rank
- Original corpus sentence
- Real source file
- Real offset
- Semantic similarity

The strongest semantic demonstration is a query that uses different words from the stored sentence but still retrieves the intended meaning in the Top 5.
