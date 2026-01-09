/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',

  // API rewrites for Docker and local development
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/:path*`,
      },
      {
        source: '/backend/:path*',
        destination: `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/:path*`,
      },
    ]
  },

  // Docker-compatible output file tracing
  outputFileTracingRoot: process.cwd(),

  // Disable TypeScript errors during build
  typescript: {
    ignoreBuildErrors: true,
  },
}

module.exports = nextConfig
