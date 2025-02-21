import { Client } from "chromadb"
import process from "process"


const client = new Client()

enum Space {
    COSINE = "cosine",
    L2 = "l2",
    INNER_PRODUCT = "inner_product"
}

type CreateEF = Record<any, any>
type UpdateEF = Record<any, any>

interface EF {
    create(...args: any): CreateEF
    update: (...args: any) => UpdateEF
    name: string
    distanceMetric: Space[]
    // constructionSchema: unknown
    // updateSchema: unknown
    generateText: (text: string) => Promise<Embedding>
    hydrateFromConfig(config: Record<any, any>): EF
}

// class OpenAIEF implements EF {}

const OpenAIEF: EF = {
    name: "openai",
    create: function (options: { model: string, openAIKeyFromEnvironmentVariable: string }) {
        return [this, options]
    },
    update: function (options: { openAIKeyFromEnvironmentVariable: string }) {
        return [this, options]
    },
    generateText(text) {
        return new Promise((resolve) => {
            resolve(text)
        })
    },
    hydrateFromConfig(config: Record<any, any>) {
        return this
    },
    distanceMetric: [Space.COSINE, Space.L2, Space.INNER_PRODUCT]
}

const chroma_embedding_functions = {
    "openai": OpenAIEF
}

process.env.OPENAI_KEY = "sk-XXXXXXXXXXXXXXXX"

const collection = client.createCollection({
    name: "test",
    config: {
        embeddingFunction: OpenAIEF.create({
            model: "text-embedding-ada-002",
            openAIKeyFromEnvironmentVariable: "MY_OPENAI_API_KEY"
        }),
        hnsw: {
            construction_ef: 100
        }
    }
})

collection.query({
    queryEmbeddings: [[0.1, 0.2, 0.3]],
    config: {
        hnsw: {
            search_ef: 100
        }
    }
})
