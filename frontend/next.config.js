/** @type {import('next').NextConfig} */
const nextConfig = {
  swcMinify: false,        // disable SWC minification
  experimental: {
    forceSwcTransforms: false,
    turbo: false            // disable Turbopack
  }
}

module.exports = nextConfig;
