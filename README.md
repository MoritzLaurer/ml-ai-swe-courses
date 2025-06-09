# Courses on key ML and software engineering topics co-created with LLMs

This repo contains: (1) a workflow and prompts for using LLMs as co-teachers for creating courses and learning new things; (2) several courses created with this workflow. 

The courses happen to be on topics related to machine learnig and software engineering, but the workflow can be adapted to any other topic. 

### Courses created so far

| Topic | Content |
|-------|---------|
| [databases](./databases/) | Database course focused on building LLM applications with PostgreSQL, psycopg3, and SQLAlchemy. Covers database fundamentals, SQL/NoSQL databases (MongoDB, Redis, Vector DBs), production deployment, and API integration with FastAPI. Uses the development of a RAG chatbot backend as the running example. |
| [jax](./jax/) | JAX fundamentals course with PyTorch comparisons throughout. Uses the tuning of small LLMs for RAG/agentic tasks as the connecting theme. Covers JAX's functional programming paradigm, transformations, and neural network implementations accessible to those knowing neither JAX nor PyTorch. |
| [pytorch](./pytorch/) | PyTorch course focused on creating and tuning small generative LLMs like GPT/Llama. Includes PyTorch-JAX comparisons. Covers PyTorch fundamentals through advanced topics with practical LLM development as the central project. |
| [reinforcement-learning](./reinforcement-learning/) | Practical RL course. Covers core RL concepts (MDPs, Q-learning, policy gradients), actor-critic methods, and Reinforcement Learning from Human Feedback (RLHF) for LLM alignment. Uses tools like Gymnasium and Stable Baselines3. |
| [ml-papers-maths](./ml-papers-maths/) | Course that helps engineers understand mathematical formulas commonly found in ML research papers. Covers mathematical notation, linear algebra, probability, calculus, transformer attention mechanisms, loss functions, and reinforcement learning formulas. Focuses on practical interpretation of complex equations in LLMs and RL contexts. |
| [typescript](./typescript/) | TypeScript course covering fundamentals to advanced concepts. Includes variables and primitives, arrays/tuples, objects, type aliases, interfaces, functions, control flow, classes, imports/exports, async/await, generics, enums, utility types, error handling, and integration patterns. |

### Motivation and workflow: 
I often use LLMs as expert tutors to rehearse existing knowledge or learn new things. When I want to dive deeper into a topic, the following workflow has turned out to work very well: 
1. Prompt an LLM to create a curriculum on a topic in a `curriculum.md` file. The curriculum should be structured in modules that can be independent files (.md files or scripts like `A0_setup.sh`, `A1_intro_to_databases.md`, `B1_sql_fundamentals_sqlite.py`). See, for example, this [teacher-prompt.md](./databases/a-teacher-prompt.md) for creating a curriculum on databases.
2. Interactively improve the `curriculum.md` manually or with the LLM.
3. Once the curriculum is finished, iteratively create each module file: Prompt the LLM to generate a first draft of a specific module file and interactively improve the module either manually or with the LLM. 

The best environment for this in my experience is [Cursor](https://www.cursor.com/en), using the LLM chat pane (instead of a chat interface in the browser). Cursor enables you to directly reference and iterate on the `curriculum.md` file and existing module files so that new modules build upon each other. It also makes restarting a chat session easier as they get too long.

I'm sharing the respective prompts, the workflow, and some resulting courses here in case they might also be useful for others. For most courses, I've added the prompts as `teacher-prompt.md` files in the respective directories. It should be quite easy to adapt this workflow to any topic you want to learn more about. I hope this is useful for some people. 






