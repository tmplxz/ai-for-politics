import { Party } from '../type/party';

export async function loadParties() {
    const basePath = '/images';
    const folders = ['cdu', 'fdp', 'gruene', 'linke', 'spd'];
    const parties = [];

    for (const folder of folders) {
        // Fetch JSON metadata
        const metadata = await fetch(`${basePath}/${folder}/metafile.json`).then(res => res.json());

        // Build image URLs manually
        const images = [1, 2, 3].map(num => `${basePath}/${folder}/image/${num}.png`);

        // Fetch text file (visual_impact_points.txt)
        const visualImpactText = await fetch(`${basePath}/${folder}/visual_impact_points.txt`).then(res => res.text());

        const party = new Party(metadata.name, metadata, images, visualImpactText);
        parties.push(party);
    }

    return parties;
}