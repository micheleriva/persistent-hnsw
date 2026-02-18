import { cosine, euclidean, innerProduct } from "../src/distances.ts";
import { generateVectors } from "./helpers/generate_vectors.ts";

const dims = [128, 256, 768, 1536];

for (const dim of dims) {
  const [a, b] = generateVectors(2, dim);

  Deno.bench(`euclidean ${dim}d`, () => {
    euclidean(a, b);
  });

  Deno.bench(`cosine ${dim}d`, () => {
    cosine(a, b);
  });

  Deno.bench(`inner_product ${dim}d`, () => {
    innerProduct(a, b);
  });
}
