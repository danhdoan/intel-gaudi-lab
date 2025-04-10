import http from 'k6/http';
import { sleep } from 'k6';


export let options = {
  vus: 200, // virtual concurrent users
  duration: '30s', // run for 10 seconds
};

export default function () {
    let payload = JSON.stringify({
        prompt: Math.random().toString(36).substring(7),
        negative_prompt: Math.random() > 0.5 ? Math.random().toString(36).substring(7) : null,
        num_inference_steps: 30,
        num_images_per_prompt:1,
        width: 512,
        height: 512,
        batch_size: 1,
        guidance_scale: 7.5,
        seed: Math.floor(Math.random() * 10000),
    });

    let params = {
        headers: { 'Content-Type': 'application/json' },
    };

    let res = http.post('http://localhost:8080/generate', payload, params);

    if (res.status !== 200) {
        console.error(`❌ Request failed with status ${res.status}`);
    }

    sleep(1);
}
